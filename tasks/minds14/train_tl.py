from transformers import TrainingArguments, Trainer, Wav2Vec2Model
from datasets import load_dataset

from segue.modeling_segue import SegueForClassification

from data_utils import *


def main():
    minds14 = load_dataset("PolyAI/minds14", "en-US")['train']
    minds14_train_set, minds14_dev_set, minds14_test_set = split_data(minds14)

    model = SegueForClassification.from_pretrained(
        "declare-lab/segue-w2v2-base",
        n_classes=14,
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
    train_set = Minds14DatasetAdapter(minds14_train_set, model.processor, conv_sizes)
    dev_set = Minds14DatasetAdapter(minds14_dev_set, model.processor, conv_sizes)
    test_set = Minds14DatasetAdapter(minds14_test_set, model.processor, conv_sizes)

    # bs 8
    training_args = TrainingArguments(
        output_dir='output/minds14',
        learning_rate=1e-2,
        num_train_epochs=60,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=2,
        logging_dir='output/minds14/log',
        warmup_ratio=0.1,
        save_strategy='epoch',
        evaluation_strategy='epoch',
        logging_strategy='epoch',
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
