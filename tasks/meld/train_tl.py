from transformers import TrainingArguments, Trainer, Wav2Vec2Model

from segue.modeling_segue import SegueForClassification
from data_utils import *

def main():
    meld_train_set = MeldDataset('data/MELD/MELD.AudioOnly', DataSplitType.TRAIN)
    meld_dev_set = MeldDataset('data/MELD/MELD.AudioOnly', DataSplitType.DEV)
    meld_test_set = MeldDataset('data/MELD/MELD.AudioOnly', DataSplitType.TEST)

    model = SegueForClassification.from_pretrained(
        "declare-lab/segue-w2v2-base",
        train_backbone=False,
        n_classes=[3, 7],
    )
    # Uncomment the next TWO lines to use w2v2 baseline encoder:
    # model.speech_encoder = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
    # model.train_backbone = False
    # restore dropout in case it was disabled during pretraining
    for layer in model.speech_encoder.encoder.layers[-1:]:
        layer.feed_forward.intermediate_dropout.p = 0.1
        layer.feed_forward.output_dropout.p = 0.1

    conv_sizes = (model.speech_encoder.config.conv_kernel, model.speech_encoder.config.conv_stride)
    train_set = MeldDatasetAdapter(meld_train_set, model.processor, conv_sizes)
    dev_set = MeldDatasetAdapter(meld_dev_set, model.processor, conv_sizes)
    test_set = MeldDatasetAdapter(meld_test_set, model.processor, conv_sizes)

    training_args = TrainingArguments(
        output_dir='output/meld',
        learning_rate=1e-3,
        num_train_epochs=20,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=1,
        logging_dir='output/meld/log',
        warmup_ratio=0.1,
        save_strategy='epoch',
        evaluation_strategy='epoch',
        logging_steps=100,
        load_best_model_at_end=True,
        label_names=['labels', 'n_speech_tokens'],
        seed=39
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
