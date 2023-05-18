from pathlib import Path
import re
from datasets import load_dataset
import torch.utils.data
from transformers import TrainingArguments
from segue.configuration_segue import SegueConfig
from segue.modeling_segue import SegueModel
from segue.processing_segue import SegueProcessor
from custom_trainer import CustomTrainer

class LibriSpeechDatasetAdapter(torch.utils.data.Dataset):
    def __init__(self, hf_librispeech, processor: SegueProcessor, compute_loss: bool):
        self.hf_librispeech = hf_librispeech
        self.processor = processor
        self.compute_loss = compute_loss

    def __getitem__(self, index):
        hf_data = self.hf_librispeech[index]
        inputs = self.processor(
            text = hf_data['text'],
            audio = hf_data['audio']['array'],
            sampling_rate = hf_data['audio']['sampling_rate'],
        )
        inputs['compute_loss'] = self.compute_loss
        return inputs
    
    def __len__(self):
        return len(self.hf_librispeech)

def main():
    hf_dev_set = torch.utils.data.ConcatDataset([
        load_dataset('librispeech_asr', split='validation.clean', cache_dir='data/hf_datasets'),
        load_dataset('librispeech_asr', split='validation.other', cache_dir='data/hf_datasets'),
    ])

    checkpoints_dir = Path("output/2023-02-08-final")
    checkpoint_range = range(120_000, 145_000)
    pattern = re.compile('checkpoint-(\d*)')
    # model = SegueModel.from_pretrained(checkpoints_dir / 'averaged')
    config = SegueConfig()
    avg_model = SegueModel(config)
    avg_model.requires_grad_(False)
    for param in avg_model.parameters():
        param.zero_()
    n_checkpoints = 0
    for checkpoint_path in checkpoints_dir.iterdir():
        match = pattern.match(checkpoint_path.name)
        if match is None or int(match.group(1)) not in checkpoint_range:
            continue
        print("Using checkpoint:", checkpoint_path.name)
        model = SegueModel.from_pretrained(checkpoint_path)
        for (param_sum, param) in zip(avg_model.parameters(), model.parameters()):
            param_sum += param
        n_checkpoints += 1
    del model
    for param in avg_model.parameters():
        param /= n_checkpoints
    avg_model.save_pretrained(checkpoints_dir / 'averaged')
    model = avg_model

    dev_set = LibriSpeechDatasetAdapter(hf_dev_set, model.processor, True)

    training_args = TrainingArguments(
        output_dir='output/2023-02-08-final',
        learning_rate=3e-5,
        num_train_epochs=10,
        evaluation_strategy='steps',
        per_device_train_batch_size=8,
        per_device_eval_batch_size=1,
        logging_dir='output/2023-02-08-final',
        warmup_steps=5000,
        save_steps=5000,
        logging_steps=100,
        eval_steps=5000,
        load_best_model_at_end=True,
    )
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        eval_dataset=dev_set,
        tokenizer=model.processor,
    )
    metrics = trainer.evaluate()
    print(metrics)

if __name__ == '__main__':
    main()
