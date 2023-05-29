from collections import defaultdict
import csv
import itertools
import pathlib
import random
import torch.utils.data
from transformers import TrainingArguments, Trainer, Wav2Vec2Model
from dataset import MoseiDataset, DataSplitType, compute_metrics

from segue.modeling_segue import SegueModel, SegueForAnalyticRegression
from data_utils import MoseiDatasetAdapter


def sample_few_shot(dataset: torch.utils.data.Dataset, inst_per_class: int, seed: int = 37):
    rand = random.Random(seed)
    instances_by_class = defaultdict(list)
    for inst in dataset:
        class_ = round(inst.label)
        instances_by_class[class_].append(inst)
    for class_, instances in instances_by_class.items():
        instances_by_class[class_] = rand.sample(sorted(instances), inst_per_class)
    return list(itertools.chain.from_iterable(instances_by_class.values()))

def n_shot_per_class(segue, mosei_train_set, mosei_dev_set, mosei_test_set, n, seed):
    if n >= 0:
        mosei_train_set = sample_few_shot(mosei_train_set, n, seed)

    model = SegueForAnalyticRegression(segue, 100)

    conv_sizes = (segue.speech_encoder.config.conv_kernel, segue.speech_encoder.config.conv_stride)
    train_set = MoseiDatasetAdapter(mosei_train_set, segue.processor, conv_sizes)
    dev_set = MoseiDatasetAdapter(mosei_dev_set, segue.processor, conv_sizes)
    test_set = MoseiDatasetAdapter(mosei_test_set, segue.processor, conv_sizes)

    training_args = TrainingArguments(
        output_dir='output/mosei',
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        save_strategy='no',
        evaluation_strategy='no',
        logging_strategy='no',
        label_names=['labels', 'n_speech_tokens'],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=dev_set,
        tokenizer=segue.processor,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    model.fit()

    eval_ = trainer.evaluate()
    pred = trainer.predict(test_set)

    return eval_, pred

def main():
    mosei_train_set = MoseiDataset('data/MOSEI', 'output/cache/MOSEI', DataSplitType.TRAIN)
    mosei_dev_set = MoseiDataset('data/MOSEI', 'output/cache/MOSEI', DataSplitType.VAL)
    mosei_test_set = MoseiDataset('data/MOSEI', 'output/cache/MOSEI', DataSplitType.TEST)

    segue = SegueModel.from_pretrained(
        "declare-lab/segue-w2v2-base",
        train_backbone=False,
    )
    # Uncomment the next TWO lines to use w2v2 baseline encoder:
    # segue.speech_encoder = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
    # segue.train_backbone = False

    result_path = pathlib.Path('output/mosei/few_shot.csv')
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['n', 'Seed', 'MAE', 'Corr', 'F1', 'Acc2', 'Acc7', 'MAE', 'Corr', 'F1', 'Acc2', 'Acc7'])

    n_values = [1, 2, 4, 8, 16, 32, 64, 128, -1]
    for n, seed in itertools.product(n_values, range(37, 40)):
        eval_, pred = n_shot_per_class(segue, mosei_train_set, mosei_dev_set, mosei_test_set, n, seed)
        pred = pred.metrics
        with open(result_path, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([
                n, seed,
                eval_['eval_mae'], eval_['eval_corr'], eval_['eval_f1'], eval_['eval_accuracy'], eval_['eval_acc7'],
                pred['test_mae'], pred['test_corr'], pred['test_f1'], pred['test_accuracy'], pred['test_acc7'],
            ])

if __name__ == '__main__':
    main()
