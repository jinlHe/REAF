import os
import sys
sys.path.append(os.path.abspath('GHS-Net_scanslice_pos_v5'))
sys.path.append(os.path.abspath(''))

import logging

from eval.zeroshot_metadata_ct_rate import PROMPTS
from eval.zeroshot_ct_rate import zero_shot as run_ct_rate

def zero_shot_eval(model, data, epoch, args, tokenizer):
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module
    if args.zeroshot_template != 'organ':
        PROMPTS["Lung nodule"] = ("Not lung nodule", "Lung nodule")
        PROMPTS["Lung opacity"] = ("Not lung opacity", "Lung opacity")

    if 'zeroshot-ct-rate' in data:
        logging.info('Starting Zero-Shot CT-RATE.')
        result = run_ct_rate(model, tokenizer, data['zeroshot-ct-rate'].dataloader, args)
        logging.info('Finished Zero-Shot CT-RATE.')
        
        # 返回18类和16类的结果
        return {
            '18_classes': result['results_18']['* mean'],
            '16_classes': result['results_16']['* mean']
        }
    
    return {}