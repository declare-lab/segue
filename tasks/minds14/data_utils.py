import random
import torch
import torchaudio
from transformers import EvalPrediction
from sklearn.metrics import accuracy_score
import numpy as np


class Minds14DatasetAdapter(torch.utils.data.Dataset):
    def __init__(self, minds14_dataset, processor, conv_sizes):
        self.minds14_dataset = minds14_dataset
        self.processor = processor
        self.conv_sizes = conv_sizes
        self.resample = torchaudio.transforms.Resample(8000, 16000)

    def __getitem__(self, index):
        instance = self.minds14_dataset[index]
        audio = instance['audio']
        assert audio['sampling_rate'] == self.resample.orig_freq
        raw_speech = self.resample(torch.from_numpy(audio['array']))
        if (l := len(raw_speech.shape)) == 2:
            raw_speech = raw_speech.mean(1)
        elif l > 2:
            assert False, f"audio data has {l} axes"
        inputs = self.processor(
            text = instance['transcription'],
            audio = raw_speech,
            sampling_rate = self.resample.new_freq,
        )
        inputs['labels'] = [instance['intent_class']]
        # del inputs['speech']
        del inputs['text']
        return inputs
    
    def __len__(self):
        return len(self.minds14_dataset)

def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    truth = np.array(p.label_ids[0]).squeeze()
    acc = accuracy_score(truth, preds)

    return {
        "accuracy": acc,
    }

def split_data(dataset):
    dataset = list(dataset)
    rand = random.Random(42)
    n = len(dataset)
    print("n =", n)
    shuffled = rand.sample(dataset, k=n)
    train_set = shuffled[:round(0.6 * n)]
    dev_set = shuffled[round(0.6 * n):round(0.8 * n)]
    test_set = shuffled[round(0.8 * n):]
    return train_set, dev_set, test_set
