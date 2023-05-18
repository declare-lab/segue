from pathlib import Path
from transformers import TrainingArguments, Trainer, Wav2Vec2Model
from dataset import MoseiDataset, DataSplitType, compute_metrics
from segue.configuration_segue import SegueConfig

from segue.modeling_segue import SegueForRegression

import re

from data_utils import *


def main():
    mosei_test_set = MoseiDataset('data/MOSEI', 'output/cache/MOSEI', DataSplitType.TEST)

    checkpoints_dir = Path("output/mosei")
    checkpoint_range = range(5300, 6300)
    pattern = re.compile('checkpoint-(\d*)')
    config = SegueConfig()
    avg_model = SegueForRegression(config)
    avg_model.requires_grad_(False)
    for param in avg_model.parameters():
        param.zero_()
    n_checkpoints = 0
    for checkpoint_path in checkpoints_dir.iterdir():
        match = pattern.match(checkpoint_path.name)
        if match is None or int(match.group(1)) not in checkpoint_range:
            continue
        print("Using checkpoint:", checkpoint_path.name)
        model = SegueForRegression.from_pretrained(checkpoint_path)
        for (param_sum, param) in zip(avg_model.parameters(), model.parameters()):
            param_sum += param
        n_checkpoints += 1
    del model
    for param in avg_model.parameters():
        param /= n_checkpoints
    avg_model.save_pretrained(checkpoints_dir / 'averaged')
    model = avg_model

    conv_sizes = (model.speech_encoder.config.conv_kernel, model.speech_encoder.config.conv_stride)
    test_set = MoseiDatasetAdapter(mosei_test_set, model.processor, conv_sizes)

    training_args = TrainingArguments(
        output_dir='output/mosei',
        learning_rate=3e-5,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=1,
        logging_dir='output/mosei/log',
        warmup_ratio=0.3,
        save_steps=100,
        logging_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        label_names=['labels', 'n_speech_tokens'],
        seed=38
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=model.processor,
        compute_metrics=compute_metrics,
    )

    pred = trainer.predict(test_set)
    print(pred.metrics)

if __name__ == '__main__':
    main()
