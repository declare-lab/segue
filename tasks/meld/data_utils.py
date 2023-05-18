from transformers import EvalPrediction
from sklearn.metrics import f1_score, accuracy_score
import torch
from dataset import *

class MeldDatasetAdapter(torch.utils.data.Dataset):
    def __init__(self, meld_dataset, processor, conv_sizes):
        self.mosei_dataset = meld_dataset
        self.processor = processor
        self.conv_sizes = conv_sizes

    def __getitem__(self, index):
        instance = self.mosei_dataset[index]
        raw_speech = instance.audio
        if (pad_size := 3280 - raw_speech.shape[0]) > 0:
            raw_speech = np.pad(raw_speech, pad_size)
        if (l := len(raw_speech.shape)) == 2:
            raw_speech = raw_speech.mean(1)
        elif l > 2:
            assert False, f"audio data has {l} axes"
        inputs = self.processor(
            audio = raw_speech,
            sampling_rate = instance.sampling_rate,
            text = instance.text,
        )
        inputs['labels'] = [(instance.sentiment + 1, instance.emotion.value)]
        # del inputs['speech']
        del inputs['text']
        return inputs
    
    def __len__(self):
        return len(self.mosei_dataset)

def compute_metrics(p: EvalPrediction):   # logits, labels
    preds = [np.argmax(label_preds, axis=1) for label_preds in p.predictions]
    preds = np.stack(preds, axis=-1)
    truth = np.array(p.label_ids[0]).squeeze()

    sent_acc = accuracy_score(truth[:,0], preds[:,0])
    sent_f1 = f1_score(truth[:,0], preds[:,0], average='weighted')
    emo_acc = accuracy_score(truth[:,1], preds[:,1])
    emo_f1 = f1_score(truth[:,1], preds[:,1], average='weighted')

    preds = np.array([str(row) for row in preds])
    truth = np.array([str(row) for row in truth])
    full_acc = accuracy_score(truth, preds)

    return {
        "sent_f1": sent_f1,
        "sent_acc": sent_acc,
        "emo_f1": emo_f1,
        "emo_acc": emo_acc,
        "full_acc": full_acc,
    }
