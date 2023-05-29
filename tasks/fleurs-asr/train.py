from typing import *
import csv
import pathlib
from model import SegueForAsr
import torch.utils.data
from transformers import TrainingArguments, Trainer, Wav2Vec2Model, EvalPrediction
from datasets import load_dataset
import evaluate
from tqdm import tqdm


class FleursDatasetAdapter(torch.utils.data.Dataset):
    def __init__(self, fleurs_dataset, processor):
        self.fleurs_dataset = fleurs_dataset
        self.processor = processor

    def __getitem__(self, index):
        hf_data = self.fleurs_dataset[index]
        inputs = self.processor(
            text = hf_data['transcription'].lower(),
            audio = hf_data['audio']['array'],
            sampling_rate = hf_data['audio']['sampling_rate'],
        )
        return inputs
    
    def __len__(self):
        return len(self.fleurs_dataset)

wer_metric = evaluate.load('wer')

def make_compute_metrics(tokenizer):
    def compute_metrics(preds: EvalPrediction) -> Dict[str, float]:
        pred_tokens = preds.predictions
        labels = preds.label_ids[0]
        print("Decoding predictions...")
        predictions = ids_to_text(pred_tokens, tokenizer, True)
        print("Decoding labels...")
        label_sents = ids_to_text(labels, tokenizer)
        print("Computing WER...")
        wer = wer_metric.compute(predictions=predictions, references=label_sents)
        return {
            'wer': wer,
        }
    return compute_metrics

def ids_to_text(ids, tokenizer, unique_consecutive=False):
    ids[ids == -100] = tokenizer.pad_token_id
    sents = []
    for sent_ids in tqdm(ids):
        if unique_consecutive:
            sent_ids = torch.unique_consecutive(torch.from_numpy(sent_ids))
        sents.append(tokenizer.decode(sent_ids, skip_special_tokens=True))
    return sents

def main():
    fleurs_train = load_dataset("google/fleurs", "en_us", split="train", cache_dir='data/hf_datasets')
    fleurs_dev = load_dataset("google/fleurs", "en_us", split="validation", cache_dir='data/hf_datasets')
    fleurs_test = load_dataset("google/fleurs", "en_us", split="test", cache_dir='data/hf_datasets')

    result_path = pathlib.Path('output/fleurs/few_shot.csv')
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['n', 'Seed', 'eval_acc', 'test_acc'])

    model = SegueForAsr.from_pretrained(
        "declare-lab/segue-w2v2-base",
    )
    # model.speech_encoder = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')

    train_set = FleursDatasetAdapter(fleurs_train, model.processor)
    dev_set = FleursDatasetAdapter(fleurs_dev , model.processor)
    test_set = FleursDatasetAdapter(fleurs_test, model.processor)

    training_args = TrainingArguments(
        output_dir='output/fleurs/head-pt',
        logging_dir='output/fleurs/head-pt/log',
        learning_rate=3e-5,
        warmup_ratio=0.1,
        num_train_epochs=100,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=2,
        save_strategy='no',
        evaluation_strategy='steps',
        eval_steps=1000,
        logging_strategy='epoch',
        label_names=['target_ids', 'target_atn_mask'],
        seed=37,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=dev_set,
        tokenizer=model.processor,
        compute_metrics=make_compute_metrics(model.processor.text_tokenizer),
    )
    trainer.train()

    eval_ = trainer.evaluate()
    print(eval_)
    pred = trainer.predict(test_set)
    pred = pred.metrics
    print(pred)

if __name__ == '__main__':
    main()
