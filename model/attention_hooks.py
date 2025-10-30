
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import gc
import warnings

class AttentionFusion:

    def __init__(self, device: str = 'cuda'):

        self.device = device
    
    def fuse_slice_attention(
        self, 
        slice_weights: List[torch.Tensor],
    ) -> torch.Tensor:

        weights = [w.to(self.device) if w.device != self.device else w for w in slice_weights]
        

        layer_weights = torch.softmax(
            torch.arange(len(weights), dtype=torch.float32, device=self.device),
            dim=0
        )
        fused = sum(w * layer_weights[i] for i, w in enumerate(weights))

        return fused
    
    def fuse_scan_attention(
        self, 
        scan_weights: List[torch.Tensor],
    ) -> torch.Tensor:

        weights = [w.to(self.device) if w.device != self.device else w for w in scan_weights]

        layer_weights = torch.softmax(
            torch.arange(len(weights), dtype=torch.float32, device=self.device),
            dim=0
        )
        fused = sum(w * layer_weights[i] for i, w in enumerate(weights))

        return fused
    
    def generate_attention_map(
        self, 
        attention_weights: torch.Tensor,
        spatial_size: Optional[Tuple[int, ...]] = (14, 14, 14),
        n_regions: int = 3,
        region_spatial_size: Optional[Tuple[int, ...]] = (3, 3, 3),
        suppression_factor: float = 0.1
    ) -> torch.Tensor:

        batch_size = attention_weights.size(0)
        
        d, h, w = spatial_size
        total_spatial_tokens = d * h * w
        region_d, region_h, region_w = region_spatial_size

        seq_len_with_cls = attention_weights.size(1)
        expected_seq_len = total_spatial_tokens + 1  # +1 for CLS token
        
        if seq_len_with_cls != expected_seq_len:
            raise ValueError(f"输入序列长度 {seq_len_with_cls} 与期望的空间尺寸不匹配 {expected_seq_len} (mode: {mode})")

        attn_map = torch.norm(attention_weights, dim=-1)
        attn_map = attn_map[:, 1:]
        cls_attn = attn_map[:, 0:1]

        attn_3d = attn_map.view(batch_size, d, h, w)

        enhanced_attention_maps = []
        
        for i in range(batch_size):
            batch_attn = attn_3d[i]

            regions_scores = []
            regions_coords = []

            d_range = range(0, d - region_d + 1, region_d)
            
            for d_start in d_range:
                for h_start in range(0, h - region_h + 1, region_h):
                    for w_start in range(0, w - region_w + 1, region_w):
                        d_end = min(d_start + region_d, d)
                        h_end = min(h_start + region_h, h)
                        w_end = min(w_start + region_w, w)
                        
                        region = batch_attn[d_start:d_end, h_start:h_end, w_start:w_end]
                        region_score = region.sum().item()
                        
                        regions_scores.append(region_score)
                        regions_coords.append((d_start, h_start, w_start, d_end, h_end, w_end))

            importance_mask = torch.ones_like(batch_attn) * suppression_factor

            if len(regions_scores) > 0:
                top_indices = torch.tensor(regions_scores).topk(min(n_regions, len(regions_scores)))[1]

                for idx in top_indices:
                    d_start, h_start, w_start, d_end, h_end, w_end = regions_coords[idx]
                    importance_mask[d_start:d_end, h_start:h_end, w_start:w_end] = 1.0
            

            enhanced_attn = batch_attn * importance_mask

            enhanced_attention_maps.append(enhanced_attn.flatten())

        result = torch.stack(enhanced_attention_maps)
        result = torch.cat((cls_attn, result), dim=1)
        
        return result
