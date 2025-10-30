import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

import math
import json
import random
import gzip
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix

from open_clip import create_model_and_transforms, get_tokenizer, get_input_dtype, build_zero_shot_classifier
from open_clip.factory import _MODEL_CONFIGS
from open_clip_train.file_utils import pt_load
from open_clip_train.precision import get_autocast

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from zeroshot_metadata_ct_rate import CLASSNAMES, ORGANS, TEMPLATES, PROMPTS

# 用于ori数据类型处理
try:
    import SimpleITK as sitk
except ImportError:
    sitk = None


def get_args_parser():
    parser = argparse.ArgumentParser('Perform Zero-shot', add_help=False)
    parser.add_argument('--model', default='vit_base_singlescan_h2_token2744', type=str)
    parser.add_argument('--model-arch-path', default='src/GHS-Net_scanslice_pos_v5', type=str,
                       help='Path to the model architecture directory containing visual_encoder module')
    parser.add_argument('--cxr-bert-path', default='~/qa44_scratch2/hjl/weights/microsoft/BiomedVLP-CXR-BERT-specialized', type=str)
    parser.add_argument('--use-cxr-bert', default=True, action='store_false')

    parser.add_argument('--lora-text', default=False, action='store_true')
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--local-files-only', default=True,
                       help='Use only local files, do not download from internet')

    parser.add_argument('--data-root', default='/mnt/sda/zdz/zdz/data/medical/report/CTRATE/valid_fixed_process')
    parser.add_argument('--zeroshot-ct-rate', default='/mnt/sda/zdz/zdz/data/medical/report/CTRATE/multi_abnormality_labels/valid_predicted_labels.csv', type=str)
    parser.add_argument('--input-info', nargs='+', default=["-1150", "350", "crop"])
    parser.add_argument('--zeroshot-template', default='both', type=str, 
                       choices=['organ', 'volume', 'both'],
                       help='Template type for zero-shot evaluation: organ, volume, or both')
    parser.add_argument('--results-dir', default='./results/ct_rate/', type=str,
                       help='Directory to save results. If not exists, will be created automatically.')
    
    parser.add_argument('--print-model-arch', default=True,
                       help='Print detailed model architecture information')

    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--workers', default=4, type=int)

    return parser


def random_seed(seed=0, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def get_data(args, preprocess_fn=None):
    class ZeroShotDataset(Dataset):
        def __init__(
            self,
            root, input_filename, input_info,
            transform=None,
        ):
            self.cts = []
            df = pd.read_csv(input_filename)
            self.data_root = root.replace("train", "valid")

            for _, row in df.iterrows():
                volume_name = row['VolumeName']
                recon = volume_name.rsplit('.', 2)[0]
            
                file_extension = '.pt.gz'

                file_path = os.path.join(
                    self.data_root,
                    recon.rsplit('_', 2)[0],
                    recon.rsplit('_', 1)[0],
                    recon + file_extension
                )

                # 若文件不存在，直接跳过，避免后续加载报错
                if not os.path.isfile(file_path):
                    print(f"[Warning] File not found, skip: {file_path}")
                    continue

                self.cts.append((file_path, row[CLASSNAMES].astype(int).tolist(), volume_name))
            
            self.input_info = (float(input_info[0]), float(input_info[1]), str(input_info[2]))
            self.transform = transform

        def __len__(self):
            return len(self.cts)
        
        def __getitem__(self, idx):
            file_path, target, volume_name = self.cts[idx]

            # stage2: 使用压缩的GZ文件（现有逻辑）
            with gzip.open(file_path, 'rb') as f:
                img = torch.load(f, weights_only=True)
            
            return file_path, img[None, ...], torch.as_tensor(target)
    

    dataset = ZeroShotDataset(
        args.data_root, args.zeroshot_ct_rate, args.input_info,
        preprocess_fn
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=None,
        drop_last=False,
    )
    return dataloader


def find_threshold(y_true, y_score):
    """
    Copy from https://github.com/alibaba-damo-academy/fvlm/blob/d768ec1546fb825fcc9ea9b3e7b2754a69f870c1/calc_metrics.py#L8C1-L8C32
    Finds the optimal threshold for binary classification based on ROC curve.

    Args:
        y_true (numpy.ndarray): True labels.
        y_score (numpy.ndarray): Predicted probabilities.

    Returns:
        float: Optimal threshold.
    """

    best_threshold = 0
    best_roc = 10000

    # Iterate over potential thresholds
    thresholds = np.linspace(0, 1, 100)
    for threshold in thresholds:
        y_pred = (y_score > threshold).astype(int)
        confusion = confusion_matrix(y_true, y_pred)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        TP_r = TP / (TP + FN)
        FP_r = FP / (FP + TN)
        curr_roc = math.sqrt(((1 - TP_r) ** 2) + (FP_r ** 2))
        if curr_roc <= best_roc:
            best_roc = curr_roc
            best_threshold = threshold

    return best_threshold


def evaluate_with_template(model, tokenizer, dataloader, template_type, device, autocast, input_dtype, save_csv=False, model_name="", results_dir=""):

    with autocast():
        classifier = {}
        for key in CLASSNAMES:
            classifier.update(
                {
                    key: build_zero_shot_classifier(
                            model,
                            tokenizer=tokenizer,
                            classnames=PROMPTS[key],
                            templates=TEMPLATES[ORGANS[key]] if template_type == 'organ' else TEMPLATES[template_type],
                            num_classes_per_batch=None, # all
                            device=device,
                            use_tqdm=False,
                        )
                }
            )

    with torch.inference_mode():
        columns = ['recon'] + CLASSNAMES
        rows = []
        labels = {key: [] for key in CLASSNAMES}
        logits = {key: [] for key in CLASSNAMES}
        preds = {key: [] for key in CLASSNAMES}
        
        for batch in tqdm(dataloader, total=len(dataloader)):
            recon, image, target = batch
            image = image.to(device=device, dtype=input_dtype)
            target = target.to(device)
            row = []
            row.append(recon[0])
            
            for idx in range(target.shape[1]):
                labels[CLASSNAMES[idx]].append(target[0, idx].cpu().float().item())
            
            with autocast():
                output = model(image=image)
                image_features = output['image_features']
                logit_scale = output['logit_scale']
                
                # predict
                for key, value in classifier.items():
                    logits_per_image = logit_scale * image_features @ value
                    logits_per_image = logits_per_image.softmax(dim=1)
                    row.append(logits_per_image[0, 1].cpu().float().item())
                    logits[key].append(logits_per_image[0, 1].cpu().float().item())
                    preds[key].append(logits_per_image.argmax(-1).cpu().float().item())

            rows.append(row)

        # 保存CSV文件（如果需要）
        # if save_csv and results_dir and model_name:
        #     os.makedirs(results_dir, exist_ok=True)
        #     df = pd.DataFrame(rows, columns=columns)
        #     csv_path = os.path.join(results_dir, f'logits_{template_type}.csv')
        #     df.to_csv(csv_path, index=False)
        #     print(f"CSV file created: {csv_path}")

        results = {key: {} for key in CLASSNAMES}
        mean_balanced_acc, mean_weighted_f1, mean_recall, mean_precision = 0., 0., 0., 0.
        for key in CLASSNAMES:
            balanced_acc = balanced_accuracy_score(np.array(labels[key]), np.array(preds[key]))
            mean_balanced_acc += balanced_acc / len(CLASSNAMES)

            weighted_f1 = f1_score(np.array(labels[key]), np.array(preds[key]), average='weighted') 
            mean_weighted_f1 += weighted_f1 / len(CLASSNAMES)

            recall = recall_score(np.array(labels[key]), np.array(preds[key])) 
            mean_recall += recall / len(CLASSNAMES)

            precision = precision_score(np.array(labels[key]), np.array(preds[key])) 
            mean_precision += precision / len(CLASSNAMES)

            results[key].update({
                'acc (balanced)': balanced_acc,
                'f1 (weighted)': weighted_f1,
                'recall': recall,
                'precision': precision,
            })
        results['mean'] = {
            'mean acc (balanced)': mean_balanced_acc,
            'mean f1 (weighted)': mean_weighted_f1,
            'mean recall': mean_recall,
            'mean precision': mean_precision,
        }

        mean_auc, mean_acc, mean_weighted_f1, mean_recall, mean_precision = 0., 0., 0., 0., 0.
        for key in CLASSNAMES:
            threshold = find_threshold(np.array(labels[key]), np.array(logits[key]))

            auc = roc_auc_score(np.array(labels[key]), np.array(logits[key])) 
            mean_auc += auc / len(CLASSNAMES)

            acc = accuracy_score(np.array(labels[key]), (np.array(logits[key]) > threshold).astype(int)) 
            mean_acc += acc / len(CLASSNAMES)

            weighted_f1 = f1_score(np.array(labels[key]), (np.array(logits[key]) > threshold).astype(int), average='weighted')
            mean_weighted_f1 += weighted_f1 / len(CLASSNAMES)

            recall = recall_score(np.array(labels[key]), (np.array(logits[key]) > threshold).astype(int)) 
            mean_recall += recall / len(CLASSNAMES)

            precision = precision_score(np.array(labels[key]), (np.array(logits[key]) > threshold).astype(int)) 
            mean_precision += precision / len(CLASSNAMES)

            results[key].update({
                'auc': auc,
                '* acc (balanced)': acc,
                '* f1 (weighted)': weighted_f1,
                '* recall': recall,
                '* precision': precision,
            })
        results['* mean'] = {
            'mean auc': mean_auc,
            '* mean acc (balanced)': mean_acc,
            '* mean f1 (weighted)': mean_weighted_f1,
            '* mean recall': mean_recall,
            '* mean precision': mean_precision,
        }

    # 计算16类别结果（去除 "Medical material" 和 "Lymphadenopathy"）
    excluded_classes = ["Medical material", "Lymphadenopathy"]
    classnames_16 = [cls for cls in CLASSNAMES if cls not in excluded_classes]
    
    # 16类别指标计算 - mean 部分
    results_16 = {key: results[key] for key in classnames_16}
    mean_balanced_acc_16, mean_weighted_f1_16, mean_recall_16, mean_precision_16 = 0., 0., 0., 0.
    for key in classnames_16:
        mean_balanced_acc_16 += results[key]['acc (balanced)'] / len(classnames_16)
        mean_weighted_f1_16 += results[key]['f1 (weighted)'] / len(classnames_16)
        mean_recall_16 += results[key]['recall'] / len(classnames_16)
        mean_precision_16 += results[key]['precision'] / len(classnames_16)
    
    results_16['mean'] = {
        'mean acc (balanced)': mean_balanced_acc_16,
        'mean f1 (weighted)': mean_weighted_f1_16,
        'mean recall': mean_recall_16,
        'mean precision': mean_precision_16,
    }
    
    # 16类别指标计算 - * mean 部分  
    mean_auc_16, mean_acc_16, mean_weighted_f1_16, mean_recall_16, mean_precision_16 = 0., 0., 0., 0., 0.
    for key in classnames_16:
        mean_auc_16 += results[key]['auc'] / len(classnames_16)
        mean_acc_16 += results[key]['* acc (balanced)'] / len(classnames_16)
        mean_weighted_f1_16 += results[key]['* f1 (weighted)'] / len(classnames_16)
        mean_recall_16 += results[key]['* recall'] / len(classnames_16)
        mean_precision_16 += results[key]['* precision'] / len(classnames_16)
    
    results_16['* mean'] = {
        'mean auc': mean_auc_16,
        '* mean acc (balanced)': mean_acc_16,
        '* mean f1 (weighted)': mean_weighted_f1_16,
        '* mean recall': mean_recall_16,
        '* mean precision': mean_precision_16,
    }
    
    return results, results_16


def zero_shot(model, tokenizer, dataloader, args):
    """
    执行零样本评估，支持单个模板或同时评估两种模板
    
    Args:
        model: 模型
        tokenizer: 分词器
        dataloader: 数据加载器
        args: 参数配置
        
    Returns:
        根据模板类型返回相应的结果格式
    """
    model.eval()
    device = torch.device(args.device)
    autocast = get_autocast('amp', device_type=device.type)
    input_dtype = get_input_dtype('amp')

    # 构建完整的结果目录路径
    # results_dir = os.path.join(args.results_dir)
    # os.makedirs(results_dir, exist_ok=True)
    
    if args.zeroshot_template == 'both':
        # 同时评估两种模板
        print("评估 organ 模板...")
        organ_results, organ_results_16 = evaluate_with_template(
            model, tokenizer, dataloader, 'organ', device, autocast, input_dtype,
            save_csv=True, model_name=args.model
        )
        
        print("评估 volume 模板...")
        volume_results, volume_results_16 = evaluate_with_template(
            model, tokenizer, dataloader, 'volume', device, autocast, input_dtype,
            save_csv=True, model_name=args.model
        )
        
        return {
            'volume_results_18': volume_results,
            'volume_results_16': volume_results_16,
            'organ_results_18': organ_results,
            'organ_results_16': organ_results_16
        }
    else:
        # 评估单个模板
        results, results_16 = evaluate_with_template(
            model, tokenizer, dataloader, args.zeroshot_template, device, autocast, input_dtype,
            save_csv=True, model_name=args.model
        )
        
        return {
            'results_18': results,
            'results_16': results_16
        }


def print_model_architecture(model, input_shape=None):
    """
    打印模型架构和每层的详细信息
    """
    print("=" * 80)
    print("模型架构详细信息")
    print("=" * 80)
    
    # 计算总参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"冻结参数数量: {total_params - trainable_params:,}")
    print("-" * 80)
    
    # 打印模型结构
    print("模型结构:")
    print(model)
    print("-" * 80)
    
    # 打印每个模块的参数信息
    print("各模块参数详情:")
    print(f"{'模块名称':<60} {'参数数量':<15} {'形状':<25} {'是否可训练':<10}")
    print("-" * 110)
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        shape = str(list(param.shape))
        trainable = "是" if param.requires_grad else "否"
        print(f"{name:<60} {param_count:<15,} {shape:<25} {trainable:<10}")
    
    print("-" * 80)
    
    # 按模块汇总参数
    print("按模块汇总参数:")
    module_params = {}
    for name, param in model.named_parameters():
        module_name = name.split('.')[0]
        if module_name not in module_params:
            module_params[module_name] = {'total': 0, 'trainable': 0}
        module_params[module_name]['total'] += param.numel()
        if param.requires_grad:
            module_params[module_name]['trainable'] += param.numel()
    
    print(f"{'模块名称':<30} {'总参数数量':<15} {'可训练参数':<15} {'冻结参数':<15}")
    print("-" * 75)
    for module_name, params in module_params.items():
        frozen = params['total'] - params['trainable']
        print(f"{module_name:<30} {params['total']:<15,} {params['trainable']:<15,} {frozen:<15,}")
    
    print("=" * 80)
    
    # 如果提供了输入形状，尝试使用torchinfo打印更详细的信息
    if input_shape is not None:
        try:
            from torchinfo import summary
            print("详细模型摘要 (使用torchinfo):")
            print("-" * 80)
            summary(model, input_size=input_shape, depth=3, col_names=["input_size", "output_size", "num_params", "trainable"])
        except ImportError:
            print("注意: 安装torchinfo可以获得更详细的模型摘要信息")
            print("安装命令: pip install torchinfo")
        except Exception as e:
            print(f"无法生成详细摘要: {e}")
    
    print("=" * 80)


def main(args):
    print(f"Running zero-shot CT-RATE evaluation...")
    print(f"Model: {args.model}")
    print(f"Model architecture path: {args.model_arch_path}")
    print(f"Use CXR-BERT: {args.use_cxr_bert}")
    print(f"Local files only: {args.local_files_only}")
    print(f"Resume from: {args.resume}")
    print(f"Print model architecture: {args.print_model_arch}")
    print("-" * 50)
    
    # 动态添加模型架构路径到sys.path并导入visual_encoder
    model_arch_path = os.path.abspath(args.model_arch_path)
    if model_arch_path not in sys.path:
        sys.path.insert(0, model_arch_path)
    
    from model import visual_encoder
    print(f"Successfully imported visual_encoder from {model_arch_path}")

    
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    if args.zeroshot_template != 'organ':
        # 原实现（保留作注释）：
        PROMPTS["Lung nodule"] = ("Not lung nodule", "Lung nodule")
        PROMPTS["Lung opacity"] = ("Not lung opacity", "Lung opacity")
        # 新实现：对齐训练语料“no X / X”风格
        # PROMPTS["Lung nodule"] = ("no lung nodule", "Lung nodule")
        # PROMPTS["Lung opacity"] = ("no lung opacity", "Lung opacity")

    random_seed(42, 0)

    # create model
    for _c in os.listdir('model/model_configs/'):
        _m, _e = os.path.splitext(_c)
        if _e.lower() == '.json':
            with open(os.path.join('model/model_configs/', _c), 'r') as f:
                model_cfg = json.load(f)
            _MODEL_CONFIGS[_m] = model_cfg
    
    # Create model without downloading pretrained weights
    if args.local_files_only:
        print("Creating model without downloading pretrained weights (pretrained='')...")
        model, _, _ = create_model_and_transforms(args.model, device=args.device, precision='amp', output_dict=True)
    else:
        print("Creating model (may download pretrained weights)...")
        model, _, _ = create_model_and_transforms(args.model, device=args.device, precision='amp', output_dict=True)
    print("-" * 50)
    
    # replace with cxr_bert
    if args.use_cxr_bert:
        from transformers import AutoModel
        if args.local_files_only:
            print("Loading CXR-BERT from local files only...")
            cxr_bert = AutoModel.from_pretrained(args.cxr_bert_path, 
                                               trust_remote_code=True).bert
        else:
            print("Loading CXR-BERT (may download if not cached)...")
            cxr_bert = AutoModel.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', 
                                               trust_remote_code=True).bert
        if args.lora_text:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=8,
                lora_alpha=8,
                target_modules=["query", "value"],
                lora_dropout=0.0,
                bias="none",
            )
            cxr_bert = get_peft_model(cxr_bert, lora_config)
            for n, p in cxr_bert.named_parameters():
                p.requires_grad = (not args.lock_text_freeze_layer_norm) if "LayerNorm" in n.split(".") else False
        cxr_bert.to(device=args.device)
        model.text.transformer = cxr_bert

    checkpoint = pt_load(args.resume, map_location='cpu')
    sd = checkpoint['state_dict']
    # sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    
    # 根据参数决定是否打印模型架构信息
    if args.print_model_arch:
        print("模型加载完成，开始打印模型架构信息...")
        try:
            # 根据模型配置推断输入形状
            input_shape = (1, 1, 112, 336, 336)  # 基于代码中的图像处理逻辑
            print_model_architecture(model, input_shape)
        except Exception as e:
            print(f"打印模型架构时出错: {e}")
            # 退化到简单的打印
            print_model_architecture(model)
    else:
        print("模型加载完成。使用 --print-model-arch 参数可以查看详细的模型架构信息。")
    
    if args.local_files_only:
        print("Loading tokenizer from local files only...")
        try:
            tokenizer = get_tokenizer(args.model, trust_remote_code=True)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            print("Tokenizer loading failed in local-only mode. You may need to download the tokenizer first.")
            raise e
    else:
        print("Loading tokenizer (may download if not cached)...")
        tokenizer = get_tokenizer(args.model, trust_remote_code=True)

    # create dataset
    data = get_data(args, None)

    # zero shot
    results = zero_shot(model, tokenizer, data, args)
    
    # 构建完整的结果目录路径（与zero_shot函数中保持一致）
    results_dir = os.path.join(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    
    if args.zeroshot_template == 'both':
        # 保存两种模板的结果
        with open(os.path.join(results_dir, 'results_volume.json'), 'w') as f:
            json.dump({
                'volume_results_18': results['volume_results_18'],
                'volume_results_16': results['volume_results_16']
            }, f, indent=4)
        
        with open(os.path.join(results_dir, 'results_organ.json'), 'w') as f:
            json.dump({
                'organ_results_18': results['organ_results_18'],
                'organ_results_16': results['organ_results_16']
            }, f, indent=4)
        
        # 保存完整结果
        with open(os.path.join(results_dir, 'results_both.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"结果已保存到:")
        print(f"  - {os.path.join(results_dir, 'results_volume.json')}")
        print(f"  - {os.path.join(results_dir, 'results_organ.json')}")
        print(f"  - {os.path.join(results_dir, 'results_both.json')}")
    else:
        # 保存单个模板的结果
        with open(os.path.join(results_dir, f'results_{args.zeroshot_template}.json'), 'w') as f:
            json.dump(results, f, indent=4)
        print(f"结果已保存到: {os.path.join(results_dir, f'results_{args.zeroshot_template}.json')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Perform Zero-shot', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
