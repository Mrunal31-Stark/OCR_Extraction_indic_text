"""Fine-tune ByT5 for Sanskrit post-OCR correction: noisy OCR line -> clean line.

Usage: python train.py --data /content/work --out /content/drive/MyDrive/sanskrit_ocr/byt5
Expects {data}/train.jsonl and {data}/dev.jsonl with fields `noisy`, `clean`.
"""
import argparse
import glob
import os

import jiwer
import numpy as np
import torch
from datasets import load_dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)

p = argparse.ArgumentParser()
p.add_argument('--data', default='/content/work')
p.add_argument('--out', default='/content/drive/MyDrive/sanskrit_ocr/byt5')
p.add_argument('--model', default='google/byt5-small')
p.add_argument('--epochs', type=float, default=1)        # 1 epoch ~= 2125 steps ~= 70 min on a T4 (1.85 s/step)
p.add_argument('--max_steps', type=int, default=-1)
p.add_argument('--bs', type=int, default=8)          # bs16 OOMs on a T4 in fp32 at 384 bytes
p.add_argument('--accum', type=int, default=4)       # effective batch 32
p.add_argument('--lr', type=float, default=5e-4)
p.add_argument('--max_len', type=int, default=384)        # p99 target = 285 bytes, max 349
p.add_argument('--eval_steps', type=int, default=500)
p.add_argument('--eval_n', type=int, default=300)         # dev lines decoded per eval (generation is slow for bytes)
args = p.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

ds = load_dataset('json', data_files={'train': f'{args.data}/train.jsonl', 'dev': f'{args.data}/dev.jsonl'})


def prep(b):
    enc = tok(b['noisy'], max_length=args.max_len, truncation=True)
    enc['labels'] = tok(text_target=b['clean'], max_length=args.max_len, truncation=True)['input_ids']
    return enc


ds = ds.map(prep, batched=True, remove_columns=ds['train'].column_names)
dev = ds['dev'].shuffle(seed=0).select(range(min(args.eval_n, len(ds['dev']))))


def compute_metrics(ev):
    preds, labels = ev
    preds = np.where(preds < 0, tok.pad_token_id, preds)
    labels = np.where(labels < 0, tok.pad_token_id, labels)
    hyp = tok.batch_decode(preds, skip_special_tokens=True)
    ref = tok.batch_decode(labels, skip_special_tokens=True)
    return {'cer': jiwer.cer(ref, hyp), 'wer': jiwer.wer(ref, hyp)}


# T4 (sm_75) has no native bf16, and T5-family models overflow to NaN in fp16 -> fp32 there; bf16 on L4/A100.
bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

targs = Seq2SeqTrainingArguments(
    output_dir=args.out,
    per_device_train_batch_size=args.bs,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=args.accum,
    learning_rate=args.lr,
    optim='adafactor',                       # T5's native optimizer; far less optimizer memory than AdamW
    lr_scheduler_type='linear',
    warmup_steps=0.03,                       # transformers>=5: float in [0,1) = ratio of total steps
    num_train_epochs=args.epochs,
    max_steps=args.max_steps,
    bf16=bf16,
    fp16=False,
    train_sampling_strategy='group_by_length',  # batch similar lengths -> less padding (bytes make padding expensive)
    eval_strategy='steps',
    eval_steps=args.eval_steps,
    save_steps=args.eval_steps,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='cer',
    greater_is_better=False,
    predict_with_generate=True,
    generation_max_length=args.max_len,
    logging_steps=50,
    dataloader_num_workers=2,
    report_to='none',
)

trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=ds['train'], eval_dataset=dev,
                         data_collator=DataCollatorForSeq2Seq(tok, model=model), processing_class=tok,
                         compute_metrics=compute_metrics)

# Resume if a checkpoint exists (Colab runtimes get recycled; checkpoints live on Drive).
last = sorted(glob.glob(f'{args.out}/checkpoint-*'), key=lambda d: int(d.rsplit('-', 1)[1]))
trainer.train(resume_from_checkpoint=last[-1] if last else None)
trainer.save_model(f'{args.out}/final')
tok.save_pretrained(f'{args.out}/final')
print('saved', f'{args.out}/final', '| best dev CER:', trainer.state.best_metric)
