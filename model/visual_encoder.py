import os
import sys

sys.path.append(os.path.abspath(''))

from functools import partial
from typing import Optional, Dict, List, Tuple, Union
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models import register_model, build_model_with_cfg, checkpoint
from timm.models.vision_transformer import VisionTransformer

from .patch_embed import PatchEmbed3D
from .pos_embed import study_pos_embed, resample_1d_posemb, resample_3d_posemb
from .attention_hooks import AttentionFusion


class REAF(VisionTransformer):
    def __init__(self, **kwargs):
        max_num_scans = kwargs.pop('max_num_scans')
        self.slice_attn_indexes = kwargs.pop('slice_attn_indexes', ())
        self.scan_attn_indexes = kwargs.pop('scan_attn_indexes', ())
        self.study_attn_indexes = kwargs.pop('study_attn_indexes', ())
        super().__init__(**kwargs)

        self._attention_fusion = AttentionFusion()
        self._extract_attention = True

        self._last_attention_weights: Optional[Dict[str, List[torch.Tensor]]] = None
        self._final_scan_embedding: Optional[torch.Tensor] = None
        self._final_slice_embedding: Optional[torch.Tensor] = None
        self.fusion_layer = nn.Linear(self.embed_dim * 2, self.embed_dim)


        # reset pos_embed
        spatial_posemb, sequential_posemb = study_pos_embed(
            max_num_scans=max_num_scans,
            grid_size=self.patch_embed.grid_size,
            embed_dim=self.embed_dim,
            pretrained_posemb=None,
        )
        self.spatial_posemb = nn.Parameter(spatial_posemb)
        self.spatial_posemb.requires_grad = False
        if sequential_posemb is not None:
            self.sequential_posemb = nn.Parameter(sequential_posemb)
            self.sequential_posemb.requires_grad = False
        else:
            self.sequential_posemb = None

    def _pos_embed(self, x):
        # x: [bs, n, d, h, w, c]
        bs, n, d, h, w, _ = x.shape
        spatial_posemb = resample_3d_posemb(self.spatial_posemb, (d, h, w), self.patch_embed.grid_size)
        if self.sequential_posemb is not None:
            sequential_posemb = resample_1d_posemb(self.sequential_posemb, n, is_train=bs != 1)
            pos_embed = sequential_posemb[:, :, None, None, None, :] + spatial_posemb[:, None, :, :, :, :]
            pos_embed = pos_embed.expand(bs, -1, -1, -1, -1, -1)
        else:
            pos_embed = spatial_posemb[:, None, :, :, :, :].expand(bs, n, -1, -1, -1, -1)

        # start status for vit blocks
        if 0 in self.slice_attn_indexes:
            pos_embed = pos_embed.flatten(3, 4).flatten(0, 2)  # [bs * n * d, h * w, c]
            x = x.flatten(3, 4).flatten(0, 2)  # [bs * n * d, h * w, c]
        elif 0 in self.scan_attn_indexes:
            pos_embed = pos_embed.flatten(2, 4).flatten(0, 1)  # [bs * n, d * h * w, c]
            x = x.flatten(2, 4).flatten(0, 1)  # [bs * n, d * h * w, c]
        elif 0 in self.study_attn_indexes:
            pos_embed = pos_embed.flatten(1, 4)  # [bs , n * d * h * w, c]
            x = x.flatten(1, 4)  # [bs , n * d * h * w, c]

        x = self.pos_drop(x + pos_embed)

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))
        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)

        return x, n, d

    def _slice2scan(self, x, num_slices):
        """
        Slice unpartition into the original scan.
        Args:
            x (tensor): input tokens with [B * num_scans * num_slices, num_prefix_tokens + L, C].
            num_slices (int): number of slices in one scan.

        Returns:
            x: [B * num_scans, num_prefix_tokens + num_slices * L, C].
        """
        prefix_tokens, src = x[:, :self.num_prefix_tokens, :].contiguous(), x[
            :, self.num_prefix_tokens:, :].contiguous()
        BND, L, C = src.shape

        prefix_tokens = prefix_tokens.view(BND // num_slices, num_slices, self.num_prefix_tokens, C).mean(dim=1)
        src = src.view(BND // num_slices, num_slices, L, C).view(BND // num_slices, num_slices * L, C)

        x = torch.cat([prefix_tokens, src], dim=1)
        return x

    def _slice2study(self, x, num_scans, num_slices):
        """
        Slices unpartition into the original study.
        Args:
            x (tensor): input tokens with [B * num_scans * num_slices, num_prefix_tokens + L, C].
            num_scans (int): number of scans in one study.
            num_slices (int): number of slices in on scan.

        Returns:
            x: [B, num_prefix_tokens + num_scans * num_slices * L, C].
        """
        prefix_tokens, src = x[:, :self.num_prefix_tokens, :].contiguous(), x[
            :, self.num_prefix_tokens:, :].contiguous()
        BND, L, C = src.shape

        prefix_tokens = prefix_tokens.view(BND // (num_scans * num_slices), num_scans * num_slices,
                                           self.num_prefix_tokens, C).mean(dim=1)
        src = src.view(BND // (num_scans * num_slices), num_scans * num_slices, L, C).view(
            BND // (num_scans * num_slices), num_scans * num_slices * L, C)

        x = torch.cat([prefix_tokens, src], dim=1)
        return x

    def _scan2study(self, x, num_scans):
        """
        Scans unpartition into the original study.
        Args:
            x (tensor): input tokens with [B * num_scans, num_prefix_tokens + L, C].
            num_scans (int): number of scans in one study.

        Returns:
            x: [B, num_prefix_tokens + num_scans * L, C].
        """
        prefix_tokens, src = x[:, :self.num_prefix_tokens, :].contiguous(), x[
            :, self.num_prefix_tokens:, :].contiguous()
        BN, L, C = src.shape

        prefix_tokens = prefix_tokens.view(BN // num_scans, num_scans, self.num_prefix_tokens, C).mean(dim=1)
        src = src.view(BN // num_scans, num_scans, L, C).view(BN // num_scans, num_scans * L, C)

        x = torch.cat([prefix_tokens, src], dim=1)
        return x

    def _scan2slice(self, x, num_slices):
        """
        Scan partition into non-overlapping slices.
        Args:
            x (tensor): input tokens with [B * num_scans, num_prefix_tokens + num_slices * L, C].
            num_slices (int): number of slices in one scan.

        Returns:
            x: [B * num_scans * num_slices, num_prefix_tokens + L, C].
        """
        prefix_tokens, src = x[:, :self.num_prefix_tokens, :].contiguous(), x[
            :, self.num_prefix_tokens:, :].contiguous()
        BN, DL, C = src.shape

        prefix_tokens = prefix_tokens.view(BN, 1, self.num_prefix_tokens, C).expand(-1, num_slices, -1, -1).contiguous()
        src = src.view(BN, num_slices, DL // (num_slices), C)

        x = torch.cat([prefix_tokens, src], dim=2)
        x = x.view(-1, self.num_prefix_tokens + DL // num_slices, C)
        return x

    def forward(self, x):
        x = self.patch_embed(x)  # [b, n, d, h, w, c]
        x, num_scans, num_slices = self._pos_embed(
            x)  # starts from: [b * n * d, h * w, c] if have slice attn else [b * n, d * h * w, c]
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        scan_weights = []
        slice_weights = []

        for idx, blk in enumerate(self.blocks):
            if idx - 1 in self.slice_attn_indexes and idx in self.scan_attn_indexes:
                x = self._slice2scan(x, num_slices)
            elif idx - 1 in self.scan_attn_indexes and idx in self.slice_attn_indexes:
                x = self._scan2slice(x, num_slices)
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x)
            else:
                x = blk(x)

            if idx in self.scan_attn_indexes:
                scan_weights.append(x)
            if idx in self.slice_attn_indexes:
                slice_weights.append(x)
        if len(self.blocks) - 1 in self.scan_attn_indexes:
            x = self._scan2study(x, num_scans)
        elif len(self.blocks) - 1 in self.slice_attn_indexes:
            x = self._slice2study(x, num_slices, num_scans)

        final_scan_embedding = self._apply_scan_attention_pooling(scan_weights)
        final_slice_embedding = self._apply_slice_attention_pooling(slice_weights)
        x = self._fuse_scan_slice_features(x, final_scan_embedding, final_slice_embedding)

        x = self.norm(x) 
        # 融合scan和slice特征
    
        x = self.forward_head(x)

        return x

    def _apply_scan_attention_pooling(
            self,
            scan_weights: List[torch.Tensor]
    ) -> torch.Tensor:

        if self._attention_fusion is not None:
            fused_scan = self._attention_fusion.fuse_scan_attention(scan_weights)

            scan_attention = self._attention_fusion.generate_attention_map(fused_scan)

            patch_embeddings = scan_weights[-1]

            attention_sum = scan_attention.sum(dim=-1)
            is_normalized = torch.allclose(attention_sum, torch.ones_like(attention_sum), atol=1e-3)

            if not is_normalized:
                scan_attention = F.softmax(scan_attention, dim=-1)

            weights_reshaped = scan_attention.unsqueeze(1)
            weighted_features = torch.bmm(weights_reshaped, patch_embeddings)

            return weighted_features.squeeze(1)


    def _apply_slice_attention_pooling(
            self,
            slice_weights: List[torch.Tensor]
    ) -> None:

        fused_slice = self._attention_fusion.fuse_slice_attention(slice_weights)

        batchsize = slice_weights[0].shape[0] // 14
        fused_slice = fused_slice.reshape(batchsize, 14, 197, 768)
        final_cls_tokens = fused_slice[:, 0, 0:1, :]
        final_patch_tokens = fused_slice[:, :, 1:, :].reshape(batchsize, 14 * 196, 768)
        fused_slice = torch.cat((final_cls_tokens, final_patch_tokens), dim=1)
        slice_attention = self._attention_fusion.generate_attention_map(fused_slice)

        patch_embeddings = slice_weights[-1]


        fused_slice = patch_embeddings.reshape(batchsize, 14, 197, 768)
        final_cls_tokens = fused_slice[:, 0, 0:1, :]
        final_patch_tokens = fused_slice[:, :, 1:, :].reshape(batchsize, 14 * 196, 768)
        patch_embeddings = torch.cat((final_cls_tokens, final_patch_tokens), dim=1)

        attention_sum = slice_attention.sum(dim=-1, keepdim=True)
        is_normalized = torch.allclose(
            attention_sum.squeeze(),
            torch.ones(slice_attention.size(0), device=slice_attention.device),
            atol=1e-3
        )

        if not is_normalized:
            slice_attention = F.softmax(slice_attention, dim=-1)

        weights_reshaped = slice_attention.unsqueeze(1)

        weighted_features = torch.bmm(weights_reshaped, patch_embeddings)

        final_slice_embedding = weighted_features.squeeze(1)  # [14, 768]

        return final_slice_embedding

    def _fuse_scan_slice_features(self, x: torch.Tensor, final_scan_embedding: torch.Tensor, final_slice_embedding: torch.Tensor) -> torch.Tensor:

        fused_embedding = torch.cat([final_scan_embedding, final_slice_embedding], dim=-1)
        final_embedding = self.fusion_layer(fused_embedding)  # [bs, 768]
        final_embedding = final_embedding.unsqueeze(1)

        return x + final_embedding


def custom_checkpoint_filter_fn(state_dict, model, patch_size=(16, 16, 16)):
    out_dict = {}
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)

    # determine whether the cls_token has corresponding pos_embed
    embed_len = state_dict['pos_embed'].shape[1]
    if torch.sqrt(torch.tensor(embed_len)) != torch.sqrt(torch.tensor(embed_len)).floor():
        out_dict['cls_token'] = state_dict.pop('cls_token') + state_dict['pos_embed'][:, 0]
        state_dict['pos_embed'] = state_dict['pos_embed'][:, 1:]

    for k, v in state_dict.items():
        if 'patch_embed' in k:
            if model.patch_embed.__class__ == PatchEmbed3D:
                if 'weight' in k:
                    if (v.shape[2], v.shape[3]) != (patch_size[1], patch_size[2]):
                        v = torch.nn.functional.interpolate(v, size=(patch_size[1], patch_size[2]), mode='bicubic')
                    v = v.sum(dim=1, keepdim=True).unsqueeze(2).repeat(1, 1, patch_size[0], 1, 1).div(patch_size[0])
            else:
                continue
        if 'pos_embed' in k:
            spatial_posemb, _ = study_pos_embed(
                max_num_scans=1,
                grid_size=model.patch_embed.grid_size,
                embed_dim=model.embed_dim,
                pretrained_posemb=v
            )
            out_dict['spatial_posemb'] = spatial_posemb
            continue
        out_dict[k] = v
    return out_dict


def custom_create_vision_transformer(variant, **kwargs):
    kwargs.pop('pretrained_cfg_overlay', None)
    return build_model_with_cfg(
        model_cls=REAF,
        variant=variant,
        pretrained_cfg_overlay=dict(first_conv=None),
        pretrained_strict=False,
        pretrained_filter_fn=partial(custom_checkpoint_filter_fn, patch_size=kwargs['patch_size']),
        **kwargs,
    )


@register_model
def vit_base_singlescan_h2_token2744(pretrained=True, **kwargs):
    model_args = dict(
        max_num_scans=1, slice_attn_indexes=(0, 1, 3, 4, 6, 7, 9, 10), scan_attn_indexes=(2, 5, 8, 11),
        img_size=(112, 336, 336), patch_size=(8, 24, 24),
        in_chans=1, depth=12, embed_dim=768, num_heads=12, num_classes=0, no_embed_class=True, pos_embed='none',
        embed_layer=PatchEmbed3D
    )
    model = custom_create_vision_transformer('vit_base_patch16_224.mae', pretrained=pretrained,
                                             **dict(model_args, **kwargs))
    return model
