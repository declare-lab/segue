from collections import defaultdict
import csv
import itertools
import pathlib
import random
from transformers import TrainingArguments, Trainer, Wav2Vec2Model

from segue.modeling_segue import SegueForClassification
from dataset import *
from data_utils import *


def sample_few_shot(dataset: MeldDataset, inst_per_class: int, seed: int = 37):
    rand = random.Random(seed)
    instances_by_class = defaultdict(list)
    for inst in dataset:
        class_ = inst.sentiment
        instances_by_class[class_].append(inst)
    for class_, instances in instances_by_class.items():
        instances_by_class[class_] = rand.sample(sorted(instances), inst_per_class)
    return list(itertools.chain.from_iterable(instances_by_class.values()))

def n_shot_per_class(meld_train_set, meld_dev_set, meld_test_set, n, seed):
    if n >= 0:
        meld_train_set = sample_few_shot(meld_train_set, n, seed)

    model = SegueForClassification.from_pretrained(
        "declare-lab/segue-w2v2-base",
        train_backbone=False,
        n_classes=[3, 7]
    )
    # Uncomment the next TWO lines to use w2v2 baseline encoder:
    # model.speech_encoder = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
    # model.train_backbone = False

    conv_sizes = (model.speech_encoder.config.conv_kernel, model.speech_encoder.config.conv_stride)
    train_set = MeldDatasetAdapter(meld_train_set, model.processor, conv_sizes)
    dev_set = MeldDatasetAdapter(meld_dev_set, model.processor, conv_sizes)
    test_set = MeldDatasetAdapter(meld_test_set, model.processor, conv_sizes)

    training_args = TrainingArguments(
        output_dir='output/meld',
        learning_rate=1e-2,
        warmup_ratio=0.1,
        num_train_epochs=5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        save_strategy='no',
        evaluation_strategy='no',
        logging_strategy='no',
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
    meld_train_set = MeldDataset('data/MELD/MELD.AudioOnly', DataSplitType.TRAIN)
    meld_dev_set = MeldDataset('data/MELD/MELD.AudioOnly', DataSplitType.DEV)
    meld_test_set = MeldDataset('data/MELD/MELD.AudioOnly', DataSplitType.TEST)

    result_path = pathlib.Path('output/meld/few_shot_sent.csv')
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['n', 'Seed', 'sent_f1', 'sent_acc'])

    n_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, -1]
    for n, seed in itertools.product(n_values, range(37, 40)):
        eval_, pred = n_shot_per_class(meld_train_set, meld_dev_set, meld_test_set, n, seed)
        pred = pred.metrics
        with open(result_path, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([
                n, seed,
                pred['test_sent_f1'], pred['test_sent_acc'],
            ])

if __name__ == '__main__':
    main()
