import os
import sys
import csv
import json
import pandas as pd
import numpy as np
import torch
import time
import random
import gzip
sys.path.append(os.path.abspath('GHS-Net_scanslice_pos_v5'))
sys.path.append(os.path.abspath(''))

# 用于ori数据类型处理
try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

from torchvision.transforms import Normalize
from open_clip_train.data import *

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

CT_RATE_INVALID_DATA = ['train_1267_a_4', 'train_11755_a_3', 'train_11755_a_4']


class StudyInfo(object):
    def __init__(self, root, volume_name, abnormalities, headers, input_data_type="stage2", report_root="", shared_report_cache=None):
        self.volume_name = volume_name
        self.abnormalities = abnormalities
        self.headers = headers
        self.report_root = report_root
        self._shared_report_cache = shared_report_cache  # 使用共享的report缓存

        recon = self.volume_name.rsplit('.', 2)[0]
        
        # 根据数据类型选择文件扩展名
        if input_data_type == "stage1":
            file_extension = '.pt'
        elif input_data_type == "stage2":
            file_extension = '.pt.gz'
        elif input_data_type == "ori":
            # ori类型直接使用原始的NIfTI文件路径
            file_extension = ''  # 保持原始的volume_name
        else:
            raise ValueError(f"Unsupported input_data_type: {input_data_type}")
            
        if input_data_type == "ori":
            # ori类型使用原始的NIfTI文件路径结构
            pt_path = os.path.join(
                root,
                recon.rsplit('_', 2)[0],
                recon.rsplit('_', 1)[0],
                self.volume_name  # 直接使用原始文件名
            )
        else:
            pt_path = os.path.join(
                root,
                # 'train',
                recon.rsplit('_', 2)[0],
                recon.rsplit('_', 1)[0],
                recon + file_extension
            )
        
        self.scans = [pt_path]  # 包装成列表
        self.scans = np.array(self.scans)

    def get_report(self, shuffle):
        return self._get_report_from_json()

    
    def _get_report_from_json(self) -> str:
        report_text = self._shared_report_cache.get(self.volume_name)
        return report_text


    def get_scans(self, shuffle):
        if shuffle: # this is for training
            return np.random.permutation(self.scans).tolist()
        else:
            return self.scans.tolist()


class StudyDataset(Dataset):
    def __init__(
        self, 
        json_root, data_root, input_filename, input_info,
        transform=None,
        tokenizer=None,
        input_data_type="stage2",
        report_root="",
    ):
        # 读取CSV文件
        csv_path = os.path.join(json_root, input_filename)
        df = pd.read_csv(csv_path)
        
        # 保存头部信息
        self.head = df.columns.tolist()
        
        # 初始化共享的报告缓存
        self.shared_report_cache = self._load_shared_report_cache(report_root)
        
        # 创建StudyInfo对象列表
        self.studies = []
        for _, row in df.iterrows():
            volume_name = row['VolumeName']
            abnormalities = row.values.tolist()  # 包含VolumeName和所有标签
            study_info = StudyInfo(root=data_root, volume_name=volume_name, 
                                 abnormalities=abnormalities, headers=self.head,
                                 input_data_type=input_data_type, report_root=report_root,
                                 shared_report_cache=self.shared_report_cache)
            self.studies.append(study_info)
        
        self.input_info = (float(input_info[0]), float(input_info[1]), str(input_info[2]))
        self.transform = transform
        self.tokenizer = tokenizer
        self.input_data_type = input_data_type
        self.data_root = data_root  # 保存数据根目录，用于ori类型

    def _load_shared_report_cache(self, report_root: str) -> dict:
        """
        加载共享的报告缓存，只在数据集初始化时加载一次
        
        Args:
            report_root: JSON报告文件路径
            
        Returns:
            包含image->report映射的字典
        """
        if not report_root or not report_root.strip():
            return {}
        
        print(f"Loading shared report cache from {report_root}...")
        try:
            # 使用更内存友好的方式加载JSON
            import gc
            
            with open(report_root, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 构建高效的查找字典：image_name -> report_text
            report_cache = {}
            for item in data:
                image = item.get('image')
                if image:
                    conversations = item.get('conversations', [])
                    for conv in conversations:
                        if conv.get('from') == 'gpt':
                            report_text = conv.get('value', '')
                            report_cache[image] = report_text
                            break
            
            # 清理原始数据以释放内存
            del data
            gc.collect()
            
            print(f"Loaded {len(report_cache)} reports into shared cache")
            return report_cache
            
        except Exception as e:
            print(f"Warning: Failed to load shared report cache from {report_root}: {e}")
            print("Will fall back to label-based report generation.")
            return {}

    def __len__(self):
        return len(self.studies)
    
    def _load_nifti_file_ori(self, volume_name: str, data_root: str) -> torch.Tensor:
        """
        加载并处理原始NIfTI文件，复用process_nii_to_final_pt.py中的逻辑
        
        Args:
            volume_name: NIfTI文件名
            data_root: 数据根目录
            
        Returns:
            处理后的图像张量
        """
        if sitk is None:
            raise ImportError("SimpleITK is required for 'ori' data type. Please install it with: pip install SimpleITK")
        
        # 构建NIfTI文件路径
        name = volume_name
        recon = name.rsplit('.', 2)[0]
        path = os.path.join(data_root, recon.rsplit('_', 2)[0], recon.rsplit('_', 1)[0], name)
        
        # 加载图像
        img = sitk.ReadImage(path)
        # 获取当前的体素间距
        current_spacing = img.GetSpacing()  # (x, y, z)
        
        # 转换为numpy数组
        img_array = sitk.GetArrayFromImage(img)  # d, h, w (z, y, x)

        # 重采样到目标间距 [3, 1, 1] - 硬编码，与process_nii_to_final_pt.py一致
        spacing = [3, 1, 1]  # [z, y, x]
        target_size = (
            int(img_array.shape[0] * current_spacing[2] / spacing[0]),  # z方向
            int(img_array.shape[1] * current_spacing[1] / spacing[1]),  # y方向  
            int(img_array.shape[2] * current_spacing[0] / spacing[2])   # x方向
        )
        
        # 使用PyTorch进行重采样
        img_tensor = torch.from_numpy(img_array).float()[None, None, ...]
        img_resampled = torch.nn.functional.interpolate(
            img_tensor, 
            size=target_size, 
            mode='trilinear',
            align_corners=False
        ).squeeze()

        return img_resampled
    
    def _process_raw_tensor(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        对原始张量进行标准化、变换和归一化处理
        
        Args:
            img_tensor: 原始图像张量
            
        Returns:
            处理后的图像张量
        """
        # 标准化到[0,1]范围
        img = (img_tensor - self.input_info[0]) / (self.input_info[1] - self.input_info[0])
        img = torch.clip(img, 0., 1.)
        img = img[None, ...].float()  # [1, d, h, w]

        # 变换处理
        if self.transform:
            img = self.transform(img)
            img = torch.as_tensor(img).float()
        else:
            if self.input_info[2] == "crop":
                # pad
                _, d, h, w = img.shape
                pad_d = max(112 - d, 0)
                pad_h = max(336 - h, 0)
                pad_w = max(336 - w, 0)
                pad_d1, pad_d2 = pad_d // 2, pad_d - pad_d // 2
                pad_h1, pad_h2 = pad_h // 2, pad_h - pad_h // 2
                pad_w1, pad_w2 = pad_w // 2, pad_w - pad_w // 2
                img = torch.nn.functional.pad(
                    img[None, ...], (pad_w1, pad_w2, pad_h1, pad_h2, pad_d1, pad_d2),
                    mode='constant',
                    value=0
                ).squeeze(0)

                # crop [hard code]: tuning this is not interesting
                _, d, h, w = img.shape
                start_d = (d - 112) // 2
                start_h = (h - 336) // 2
                start_w = (w - 336) // 2
                img = img[
                    :,
                    start_d:start_d + 112,
                    start_h:start_h + 336,
                    start_w:start_w + 336
                ]

            elif self.input_info[2] == "resize":
                img = torch.nn.functional.interpolate(img[None, ...], size=(112, 336, 336), mode='trilinear').squeeze(0)

            else:
                raise NotImplementedError(f"不支持的变换类型: {self.input_info[2]}")

        # normalize
        normalizer = Normalize(torch.as_tensor(IMAGENET_DEFAULT_MEAN).mean(), torch.as_tensor(IMAGENET_DEFAULT_STD).mean())
        img = normalizer(img)
        
        return img
    
    def _get_item(self, idx):
        """获取单个数据项的内部方法"""
        study = self.studies[idx]

        # get report
        # print("DEBUG report:",study.get_report(shuffle=True))
        report = self.tokenizer([str(study.get_report(shuffle=True))])[0]

        # get scan
        scan = study.get_scans(shuffle=True)[0] # CT-RATE is a curated dataset
        
        # 根据input_data_type选择数据加载方式
        if self.input_data_type == "stage1":
            # stage1: 使用未压缩的PT文件
            img_tensor = torch.load(scan, weights_only=True)
            # 使用公共处理函数
            img = self._process_raw_tensor(img_tensor)
        
        elif self.input_data_type == "stage2":
            # stage2: 使用压缩的GZ文件（现有逻辑）
            with gzip.open(scan, 'rb') as f:
                img = torch.load(f, weights_only=True)
        
        elif self.input_data_type == "ori":
            # ori类型: 从原始NIfTI文件处理
            volume_name = study.volume_name
            # 使用_load_nifti_file_ori加载并重采样NIfTI文件
            img_tensor = self._load_nifti_file_ori(volume_name, self.data_root)
            # 使用公共处理函数
            img = self._process_raw_tensor(img_tensor)
        
        else:
            raise ValueError(f"Unsupported input_data_type: {self.input_data_type}")

        return img[None, ...], report

    def __getitem__(self, idx):
        """获取数据项，包含重试机制"""
        torch.cuda.empty_cache()
        num_base_retries = 2
        num_final_retries = 2

        # 首先尝试当前样本
        for attempt_idx in range(num_base_retries):
            try:
                # print("DEBUG idx:", idx)
                sample = self._get_item(idx)
                return sample
            except Exception as e:
                # 如果是云盘问题，等待1秒
                print(f"[Base try #{attempt_idx}] Failed to fetch sample {idx}. Exception:", str(e)[:100])
                time.sleep(1)

        # 尝试其他样本，以防文件损坏问题
        for attempt_idx in range(num_base_retries):
            try:
                # 尝试几种不同的策略来找到替代样本
                if attempt_idx == 0:
                    next_index = min(idx + 1, len(self.studies) - 1)
                elif attempt_idx == 1:
                    next_index = max(idx - 1, 0)
                else:
                    next_index = random.choice(range(len(self)))
                
                sample = self._get_item(next_index)
                print(f"[Try other #{attempt_idx}] Successfully fetched alternative sample {next_index} instead of {idx}")
                return sample
            except Exception as e:
                # 不需要等待
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    str(e)[:100],
                )
                pass

        # 最后的绝望尝试，随机采样
        for attempt_idx in range(num_final_retries):
            try:
                random_index = random.choice(range(len(self)))
                sample = self._get_item(random_index)
                print(f"[Final try #{attempt_idx}] Successfully fetched random sample {random_index} instead of {idx}")
                return sample
            except Exception as e:
                print(f"[Final try #{attempt_idx}] Failed to fetch random sample. Exception:", str(e)[:100])
                continue

        # 如果所有尝试都失败，抛出最后一个异常
        raise RuntimeError(f"Failed to fetch any valid sample after {num_base_retries + num_base_retries + num_final_retries} attempts. Original index: {idx}")


def get_train_dataset(args, preprocess_fn, tokenizer=None):
    input_filename = args.train_data
    assert input_filename
    dataset = StudyDataset(
        args.json_root, args.data_root, input_filename,
        args.input_info,
        preprocess_fn,
        tokenizer,
        input_data_type=getattr(args, 'input_data_type', 'stage2'),
        report_root=getattr(args, 'report_root', ''),
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed else None
    shuffle = sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=True,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_zeroshot_ct_rate_dataset(args, preprocess_fn):
    from eval.zeroshot_ct_rate import get_data
    dataloader = get_data(args, preprocess_fn)
    dataloader.num_samples = len(dataloader.dataset)
    dataloader.num_batches = len(dataloader)
    return DataInfo(dataloader, None)


def get_data(args, tokenizer=None):
    data = {}
    if args.train_data:
        data["train"] = get_train_dataset(args, None, tokenizer=tokenizer)
    if args.zeroshot_ct_rate:
        data["zeroshot-ct-rate"] = get_zeroshot_ct_rate_dataset(args, None)
    return data
