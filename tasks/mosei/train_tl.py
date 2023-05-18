from transformers import TrainingArguments, Trainer, Wav2Vec2Model
from dataset import MoseiDataset, DataSplitType, compute_metrics

from segue.modeling_segue import SegueForRegression
from .data_utils import *

def main():
    mosei_train_set = MoseiDataset('data/MOSEI', 'output/cache/MOSEI', DataSplitType.TRAIN)
    mosei_dev_set = MoseiDataset('data/MOSEI', 'output/cache/MOSEI', DataSplitType.VAL)
    mosei_test_set = MoseiDataset('data/MOSEI', 'output/cache/MOSEI', DataSplitType.TEST)

    model = SegueForRegression.from_pretrained(
        "output/2023-02-08-final/averaged-10",
        train_backbone=False,
    )
    # Uncomment the next TWO lines to use w2v2 baseline encoder:
    # model.speech_encoder = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
    # model.train_backbone = False
    # restore dropout in case it was disabled during pretraining
    for layer in model.speech_encoder.encoder.layers[-1:]:
        layer.feed_forward.intermediate_dropout.p = 0.1
        layer.feed_forward.output_dropout.p = 0.1

    conv_sizes = (model.speech_encoder.config.conv_kernel, model.speech_encoder.config.conv_stride)
    train_set = MoseiDatasetAdapter(mosei_train_set, model.processor, conv_sizes)
    dev_set = MoseiDatasetAdapter(mosei_dev_set, model.processor, conv_sizes)
    test_set = MoseiDatasetAdapter(mosei_test_set, model.processor, conv_sizes)

    training_args = TrainingArguments(
        output_dir='output/mosei',
        learning_rate=3e-5,
        num_train_epochs=20,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=1,
        logging_dir='output/mosei/log',
        warmup_ratio=0.1,
        save_strategy='epoch',
        evaluation_strategy='epoch',
        logging_steps=100,
        load_best_model_at_end=True,
        label_names=['labels', 'n_speech_tokens'],
        seed=37
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
    pred = trainer.predict(test_set)
    print(pred.metrics)

if __name__ == '__main__':
    main()
