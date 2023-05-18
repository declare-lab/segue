from sklearn.metrics import accuracy_score
from tasks.fsc.dataset import FluentSpeechCommandsDataset
from transformers import EvalPrediction
import torch
import numpy as np

class FSCDatasetAdapter(torch.utils.data.Dataset):
    def __init__(self, fsc_dataset: FluentSpeechCommandsDataset, processor, conv_sizes):
        self.fsc_dataset = fsc_dataset
        self.processor = processor
        self.conv_sizes = conv_sizes
    
    def __getitem__(self, index):
        instance = self.fsc_dataset[index]
        audio = instance.audio
        raw_speech = torch.from_numpy(audio)
        if (l := len(raw_speech.shape)) == 2:
            raw_speech = raw_speech.mean(1)
        elif l > 2:
            assert False, f"audio data has {l} axes"
        inputs = self.processor(
            text = instance.transcription,
            audio = raw_speech,
            sampling_rate = instance.sampling_rate,
        )
        inputs['labels'] = [(
            FluentSpeechCommandsDataset.actions.index(instance.action),
            FluentSpeechCommandsDataset.objs.index(instance.obj),
            FluentSpeechCommandsDataset.locations.index(instance.location),
        )]
        # del inputs['speech']
        del inputs['text']
        return inputs
    
    def __len__(self):
        return len(self.fsc_dataset)

def compute_metrics(p: EvalPrediction):
    preds = [np.argmax(label_preds, axis=1) for label_preds in p.predictions]
    preds = np.stack(preds, axis=-1)
    truth = np.array(p.label_ids[0]).squeeze()

    action_acc = accuracy_score(truth[:,0], preds[:,0])
    obj_acc = accuracy_score(truth[:,1], preds[:,1])
    location_acc = accuracy_score(truth[:,2], preds[:,2])

    preds = np.array([str(row) for row in preds])
    truth = np.array([str(row) for row in truth])
    full_acc = accuracy_score(truth, preds)

    return {
        "action_acc": action_acc,
        "obj_acc": obj_acc,
        "location_acc": location_acc,
        "full_acc": full_acc,
    }
