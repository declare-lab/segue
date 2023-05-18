from pathlib import Path
import pathlib
from transformers import TrainingArguments, Trainer
from segue.configuration_segue import SegueConfig

from segue.modeling_segue import SegueForClassification

import re
from tasks.fsc.dataset import DataSplitType, FluentSpeechCommandsDataset

from tasks.fsc.data_utils import FSCDatasetAdapter, compute_metrics

def main():
    fsc_path = pathlib.Path('data/fluent_speech_commands')
    fsc_dev_set = FluentSpeechCommandsDataset(fsc_path, DataSplitType.VAL)
    fsc_test_set = FluentSpeechCommandsDataset(fsc_path, DataSplitType.TEST)

    checkpoints_dir = Path("output/fsc")
    checkpoint_range = range(9_500, 14_500)
    pattern = re.compile('checkpoint-(\d*)')
    config = SegueConfig(
        n_classes=[6, 14, 8],
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
    dev_set = FSCDatasetAdapter(fsc_dev_set, model.processor, conv_sizes)
    test_set = FSCDatasetAdapter(fsc_test_set, model.processor, conv_sizes)

    training_args = TrainingArguments(
        output_dir='output/fsc',
        learning_rate=3e-5,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=1,
        logging_dir='output/fsc/log',
        warmup_ratio=0.3,
        save_steps=100,
        logging_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        label_names=['labels', 'n_speech_tokens'],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=model.processor,
        compute_metrics=compute_metrics,
    )

    pred = trainer.predict(dev_set)
    print(pred.metrics)

    pred = trainer.predict(test_set)
    print(pred.metrics)

if __name__ == '__main__':
    main()
