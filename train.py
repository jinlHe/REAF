import os
import sys
sys.path.append(os.path.abspath('GHS-Net_scanslice_pos_v5'))
sys.path.append(os.path.abspath(''))

import json
import logging
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype
from open_clip_train.distributed import is_master
from open_clip_train.precision import get_autocast

from zero_shot import zero_shot_eval

# 全局训练时间跟踪器
class GlobalTrainingTimer:
    """跟踪从训练开始到现在的总体时间"""
    
    def __init__(self):
        self.start_time = None
        self.total_batches_completed = 0
        self.total_time_elapsed = 0.0
    
    def start_training(self):
        """开始训练时调用"""
        self.start_time = time.time()
        self.total_batches_completed = 0
        self.total_time_elapsed = 0.0
    
    def update_batch(self, batch_time: float):
        """更新batch完成时间"""
        self.total_batches_completed += 1
        self.total_time_elapsed += batch_time
    
    def get_average_batch_time(self) -> float:
        """获取平均batch时间"""
        if self.total_batches_completed == 0:
            return 0.0
        return self.total_time_elapsed / self.total_batches_completed
    
    def get_total_training_time(self) -> float:
        """获取总训练时间"""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

# 创建全局训练时间跟踪器实例
global_training_timer = GlobalTrainingTimer()

# 全局最佳指标跟踪器
class BestMetricsTracker:
    """跟踪每个指标的最佳值和对应的epoch"""
    
    def __init__(self):
        self.best_metrics = {}
        self.best_epochs = {}
    
    def update(self, metrics: dict, epoch: int):
        """更新最佳指标"""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                # 对于大部分指标，值越大越好
                # 对于包含"loss"的指标，值越小越好
                if "loss" in key.lower():
                    if key not in self.best_metrics or value < self.best_metrics[key]:
                        self.best_metrics[key] = value
                        self.best_epochs[key] = epoch
                else:
                    if key not in self.best_metrics or value > self.best_metrics[key]:
                        self.best_metrics[key] = value
                        self.best_epochs[key] = epoch
    
    def get_best_summary(self) -> dict:
        """获取最佳指标总结"""
        summary = {}
        for key in self.best_metrics:
            summary[f"best {key}"] = {
                "value": self.best_metrics[key],
                "epoch": self.best_epochs[key]
            }
        return summary

# 创建全局最佳指标跟踪器实例
best_metrics_tracker = BestMetricsTracker()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // (args.accum_freq * args.accum_batch)
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    # 如果是第一个epoch，启动全局训练计时器
    if epoch == 0:
        global_training_timer.start_training()

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}
    images, texts = [], []

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, mini_batch in enumerate(dataloader):
        i_accum = i // (args.accum_freq * args.accum_batch)
        step = num_batches_per_epoch * epoch + i_accum
        
        if not args.skip_scheduler:
            scheduler(step)

        # Ensure images and texts are lists at the start of each accum_batch cycle
        if i % args.accum_batch == 0:
            images, texts = [], []

        _images, _texts = mini_batch
        images.append(_images.to(device=device, dtype=input_dtype, non_blocking=True))
        texts.append(_texts.to(device=device, non_blocking=True))

        if ((i + 1) % args.accum_batch) > 0:
            continue
        images = torch.cat(images, dim=0); texts = torch.cat(texts, dim=0)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                model_out = model(images, texts)
                logit_scale = model_out["logit_scale"]
                if args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(images, texts)
                    model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})

                losses = loss(**model_out, output_dict=True)
                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images); accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]; texts=accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                        
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time = time.time() - end
        batch_time_m.update(batch_time)
        
        # 更新全局训练计时器
        global_training_timer.update_batch(batch_time)
        
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # Calculate remaining time for training based on global average batch time
            # Calculate total steps for current epoch
            remaining_batches_in_epoch = num_batches_per_epoch - batch_count
            # Calculate total remaining epochs (current epoch is 0-indexed, so args.epochs - epoch - 1)
            remaining_epochs = args.epochs - epoch - 1
            # Calculate total remaining batches
            total_remaining_batches = remaining_batches_in_epoch + (remaining_epochs * num_batches_per_epoch)
            # Calculate remaining time based on global average batch time
            avg_batch_time = global_training_timer.get_average_batch_time()
            remaining_time_seconds = total_remaining_batches * avg_batch_time
            # Convert to hours, minutes, seconds
            remaining_hours = int(remaining_time_seconds // 3600)
            remaining_minutes = int((remaining_time_seconds % 3600) // 60)
            remaining_seconds = int(remaining_time_seconds % 60)
            
            # Calculate overall progress
            total_batches = args.epochs * num_batches_per_epoch
            completed_batches = epoch * num_batches_per_epoch + batch_count
            overall_progress = 100.0 * completed_batches / total_batches
            
            # Format remaining time string
            if remaining_hours > 0:
                eta_str = f"ETA: {remaining_hours}h {remaining_minutes}m {remaining_seconds}s"
            elif remaining_minutes > 0:
                eta_str = f"ETA: {remaining_minutes}m {remaining_seconds}s"
            else:
                eta_str = f"ETA: {remaining_seconds}s"
            
            # Add overall progress info with global average batch time
            total_training_time = global_training_timer.get_total_training_time()
            total_training_hours = int(total_training_time // 3600)
            total_training_minutes = int((total_training_time % 3600) // 60)
            total_training_seconds = int(total_training_time % 60)
            
            if total_training_hours > 0:
                elapsed_str = f"Elapsed: {total_training_hours}h {total_training_minutes}m {total_training_seconds}s"
            elif total_training_minutes > 0:
                elapsed_str = f"Elapsed: {total_training_minutes}m {total_training_seconds}s"
            else:
                elapsed_str = f"Elapsed: {total_training_seconds}s"
            
            avg_batch_time_ms = avg_batch_time * 1000
            progress_str = f"Overall: {overall_progress:.1f}% ({completed_batches}/{total_batches} batches) | Avg Batch: {avg_batch_time_ms:.1f}ms | {elapsed_str}"

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
                f"{eta_str} | {progress_str} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }            
            log_data.update({name:val.val for name,val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)
            
            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)
            
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
        
        # reset batch accum
        images, texts = [], []
    # end for


def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    
    # 处理18类和16类的结果
    if '18_classes' in zero_shot_metrics:
        # 添加18类结果（原有的指标名称）
        metrics.update(zero_shot_metrics['18_classes'])
        
        # 添加16类结果（使用前缀区分）
        for key, value in zero_shot_metrics['16_classes'].items():
            metrics[f'16cls_{key}'] = value

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_clip_score = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)
                
                with autocast():
                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()

                    # compute clip score
                    clip_scores_per_image = torch.clamp(image_features @ text_features.t(), min=0) * 100
                    total_clip_scores = clip_scores_per_image.trace()
                    batch_size = images.shape[0]

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_clip_score += total_clip_scores * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Score: {cumulative_clip_score / num_samples:.6f}\t"
                    )

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            clip_score = cumulative_clip_score / num_samples
            metrics.update(
                {**val_metrics, "clip_val_score":clip_score.item(), "epoch": epoch, "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        # 在metrics中添加epoch信息
        metrics_with_epoch = {"epoch": epoch}
        metrics_with_epoch.update(metrics)
        
        # 更新最佳指标跟踪器
        best_metrics_tracker.update(metrics, epoch)
        
        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics_with_epoch))
            f.write("\n")
            
            # 如果是最后一个epoch，写入最佳指标总结
            if epoch == args.epochs:
                best_summary = best_metrics_tracker.get_best_summary()
                
                # 分别处理18类和16类的结果
                metrics_18_cls = {}
                metrics_16_cls = {}
                
                for metric_name, best_info in best_summary.items():
                    if "16cls_" in metric_name:
                        # 16类指标，去除前缀
                        clean_name = metric_name.replace("best 16cls_", "best ")
                        metrics_16_cls[clean_name] = best_info
                    else:
                        # 18类指标
                        metrics_18_cls[metric_name] = best_info
                
                # 写入18类最佳指标总结
                f.write(json.dumps({"summary_header": "=== Best Results (18 classes) ==="}))
                f.write("\n")
                for metric_name, best_info in metrics_18_cls.items():
                    summary_line = {
                        "summary_18cls": f"{metric_name}: {best_info['value']:.4f}, epoch {best_info['epoch']}"
                    }
                    f.write(json.dumps(summary_line))
                    f.write("\n")
                
                # 写入16类最佳指标总结
                f.write(json.dumps({"summary_header": "=== Best Results (16 classes) ==="}))
                f.write("\n")
                for metric_name, best_info in metrics_16_cls.items():
                    summary_line = {
                        "summary_16cls": f"{metric_name}: {best_info['value']:.4f}, epoch {best_info['epoch']}"
                    }
                    f.write(json.dumps(summary_line))
                    f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // (args.accum_freq * args.accum_batch)
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
