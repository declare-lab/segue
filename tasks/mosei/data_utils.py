import torch
from dataset import *

class MoseiDatasetAdapter(torch.utils.data.Dataset):
    def __init__(self, mosei_dataset: MoseiDataset, processor, conv_sizes):
        self.mosei_dataset = mosei_dataset
        self.processor = processor
        self.conv_sizes = conv_sizes

    def __getitem__(self, index):
        mosei_segment = self.mosei_dataset[index]
        raw_speech = mosei_segment.audio
        if (l := len(raw_speech.shape)) == 2:
            raw_speech = raw_speech.mean(1)
        elif l > 2:
            assert False, f"audio data has {l} axes"
        inputs = self.processor(
            audio = raw_speech,
            sampling_rate = mosei_segment.sampling_rate,
            text = mosei_segment.text,
        )
        inputs['labels'] = [mosei_segment.label]
        # del inputs['speech']
        del inputs['text']
        return inputs
    
    def __len__(self):
        return len(self.mosei_dataset)

    def compute_n_tokens(self, n_samples: int) -> int:
        n = n_samples
        for kernel_size, stride in zip(*self.conv_sizes):
            n = (n - (kernel_size - stride)) // stride
        return n
