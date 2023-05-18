from transformers import TrainingArguments, Trainer, Wav2Vec2Model
from dataset import *

from segue.modeling_segue import SegueForClassification

from tasks.fsc.data_utils import *


def main():
    fsc_path = pathlib.Path('data/fluent_speech_commands')
    fsc_train_set = FluentSpeechCommandsDataset(fsc_path, DataSplitType.TRAIN)
    fsc_dev_set = FluentSpeechCommandsDataset(fsc_path, DataSplitType.VAL)
    fsc_test_set = FluentSpeechCommandsDataset(fsc_path, DataSplitType.TEST)

    model = SegueForClassification.from_pretrained(
        "output/2023-02-08-final/averaged-10",
        n_classes=[6, 14, 8],
    )
    # Uncomment to use w2v2 baseline encoder:
    # model.speech_encoder = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
    # restore dropout in case it was disabled during pretraining
    for layer in model.speech_encoder.encoder.layers[-1:]:
        layer.feed_forward.intermediate_dropout.p = 0.1
        layer.feed_forward.output_dropout.p = 0.1

    conv_sizes = (model.speech_encoder.config.conv_kernel, model.speech_encoder.config.conv_stride)
    train_set = FSCDatasetAdapter(fsc_train_set, model.processor, conv_sizes)
    dev_set = FSCDatasetAdapter(fsc_dev_set, model.processor, conv_sizes)
    test_set = FSCDatasetAdapter(fsc_test_set, model.processor, conv_sizes)

    # bs: 8
    training_args = TrainingArguments(
        output_dir='output/fsc',
        learning_rate=3e-5,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=4,
        logging_dir='output/fsc/log',
        warmup_ratio=0.1,
        save_steps=500,
        eval_steps=500,
        evaluation_strategy='steps',
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
