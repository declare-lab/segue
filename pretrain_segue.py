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
    hf_train_set = torch.utils.data.ConcatDataset([
        load_dataset('librispeech_asr', split='train.clean.100', cache_dir='data/hf_datasets'),
        load_dataset('librispeech_asr', split='train.clean.360', cache_dir='data/hf_datasets'),
        load_dataset('librispeech_asr', split='train.other.500', cache_dir='data/hf_datasets'),
    ])
    hf_dev_set = torch.utils.data.ConcatDataset([
        load_dataset('librispeech_asr', split='validation.clean', cache_dir='data/hf_datasets'),
        load_dataset('librispeech_asr', split='validation.other', cache_dir='data/hf_datasets'),
    ])

    config = SegueConfig()
    model = SegueModel(config)

    train_set = LibriSpeechDatasetAdapter(hf_train_set, model.processor, True)
    dev_set = LibriSpeechDatasetAdapter(hf_dev_set, model.processor, True)

    training_args = TrainingArguments(
        output_dir='output/2023-02-08-final',
        learning_rate=3e-5,
        num_train_epochs=10,
        evaluation_strategy='steps',
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        logging_dir='output/2023-02-08-final/log',
        warmup_steps=5000,
        save_steps=5000,
        logging_steps=100,
        eval_steps=5000,
        load_best_model_at_end=True,
    )
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=dev_set,
        tokenizer=model.processor,
    )

    trainer.train(resume_from_checkpoint=False)


if __name__ == '__main__':
    main()
