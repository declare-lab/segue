from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from pathlib import Path
import re

from segue.modeling_segue import SegueForClassification, SegueConfig

from data_utils import *


def main():
    minds14 = load_dataset("PolyAI/minds14", "en-US")['train']
    minds14_train_set, minds14_dev_set, minds14_test_set = split_data(minds14)

    checkpoints_dir = Path("output/minds14")
    checkpoint_range = range(421, 841)
    pattern = re.compile('checkpoint-(\d*)')
    config = SegueConfig(
        n_classes=14,
    )
    avg_model = SegueForClassification(config)
    avg_model.requires_grad_(False)
    for param in avg_model.parameters():
        param.zero_()
    n_checkpoints = 0
    for checkpoint_path in checkpoints_dir.iterdir():
        match = pattern.match(checkpoint_path.name)
        if match is None or int(match.group(1)) not in checkpoint_range:
            continue
        print("Using checkpoint:", checkpoint_path.name)
        model = SegueForClassification.from_pretrained(checkpoint_path)
        for (param_sum, param) in zip(avg_model.parameters(), model.parameters()):
            param_sum += param
        n_checkpoints += 1
    del model
    for param in avg_model.parameters():
        param /= n_checkpoints
    avg_model.save_pretrained(checkpoints_dir / 'averaged')
    model = avg_model

    conv_sizes = (model.speech_encoder.config.conv_kernel, model.speech_encoder.config.conv_stride)
    train_set = Minds14DatasetAdapter(minds14_train_set, model.processor, conv_sizes)
    dev_set = Minds14DatasetAdapter(minds14_dev_set, model.processor, conv_sizes)
    test_set = Minds14DatasetAdapter(minds14_test_set, model.processor, conv_sizes)

    # bs: 8
    training_args = TrainingArguments(
        output_dir='output/minds14',
        learning_rate=3e-5,
        num_train_epochs=20,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=2,
        logging_dir='output/log',
        warmup_ratio=0.1,
        save_strategy='epoch',
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        load_best_model_at_end=True,
        label_names=['labels', 'n_speech_tokens'],
        seed=38
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=dev_set,
        tokenizer=model.processor,
        compute_metrics=compute_metrics,
    )

    pred = trainer.predict(dev_set)
    print(pred.metrics)

    pred = trainer.predict(test_set)
    print(pred.metrics)

if __name__ == '__main__':
    main()
