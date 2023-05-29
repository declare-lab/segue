from collections import defaultdict
import csv
import itertools
import pathlib
import random
import torch.utils.data
from transformers import TrainingArguments, Trainer, Wav2Vec2Model
from datasets import load_dataset

from segue.modeling_segue import SegueForClassification
from data_utils import *


def sample_few_shot(dataset: torch.utils.data.Dataset, inst_per_class: int, seed: int = 37):
    rand = random.Random(seed)
    instances_by_class = defaultdict(list)
    for inst in dataset:
        class_ = inst['intent_class']
        instances_by_class[class_].append(inst)
    for class_, instances in instances_by_class.items():
        instances_by_class[class_] = rand.sample(sorted(instances, key=lambda i: i['path']), inst_per_class)
    return list(itertools.chain.from_iterable(instances_by_class.values()))

def n_shot_per_class(checkpoint_name, minds14_train_set, minds14_dev_set, minds14_test_set, n, seed):
    if n >= 0:
        minds14_train_set = sample_few_shot(minds14_train_set, n, seed)

    model = SegueForClassification.from_pretrained(
        checkpoint_name,
        n_classes=14,
        train_backbone=False,
    )
    # Uncomment the next TWO lines to use w2v2 baseline encoder:
    # model.speech_encoder = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
    # model.train_backbone = False

    conv_sizes = (model.speech_encoder.config.conv_kernel, model.speech_encoder.config.conv_stride)
    train_set = Minds14DatasetAdapter(minds14_train_set, model.processor, conv_sizes)
    dev_set = Minds14DatasetAdapter(minds14_dev_set, model.processor, conv_sizes)
    test_set = Minds14DatasetAdapter(minds14_test_set, model.processor, conv_sizes)

    training_args = TrainingArguments(
        output_dir='output/minds14',
        logging_dir='output/minds14/log',
        learning_rate=1e-2,
        warmup_ratio=0.1,
        num_train_epochs=5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        save_strategy='no',
        evaluation_strategy='no',
        logging_strategy='epoch',
        label_names=['labels', 'n_speech_tokens'],
        seed=seed
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=dev_set,
        tokenizer=model.processor,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    eval_ = trainer.evaluate()
    pred = trainer.predict(test_set)

    return eval_, pred

def main():
    minds14 = load_dataset("PolyAI/minds14", "en-US")['train']
    minds14_train_set, minds14_dev_set, minds14_test_set = split_data(minds14)

    result_path = pathlib.Path('output/minds14/few_shot.csv')
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['n', 'Seed', 'eval_acc', 'test_acc'])

    n_values = [1, 2, 4, 8, 16, -1]
    for n, seed in itertools.product(n_values, range(37, 40)):
        eval_, pred = n_shot_per_class("declare-lab/segue-w2v2-base", minds14_train_set, minds14_dev_set, minds14_test_set, n, seed)
        pred = pred.metrics
        with open(result_path, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([n, seed, eval_['eval_accuracy'], pred['test_accuracy']])

if __name__ == '__main__':
    main()
