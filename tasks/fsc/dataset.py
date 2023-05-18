import csv
from dataclasses import dataclass
from enum import Enum
import itertools
import pathlib
from typing import Optional
from torch.utils.data import Dataset
import numpy as np
import soundfile

class DataSplitType(Enum):
    TRAIN = 'train'
    VAL = 'valid'
    TEST = 'test'

@dataclass
class FluentSpeechCommandsInstance:
    index: int
    audio_path: pathlib.Path
    transcription: str
    action: str
    obj: str
    location: str

    _audio: Optional[np.ndarray] = None
    _sampling_rate: Optional[int] = None

    def _ensure_audio_loaded(self):
        if self._audio is None or self._sampling_rate is None:
            self._audio, self._sampling_rate = soundfile.read(self.audio_path)
    
    @property
    def audio(self) -> np.ndarray:
        self._ensure_audio_loaded()
        return self._audio

    @property
    def sampling_rate(self):
        self._ensure_audio_loaded()
        return self._sampling_rate

class FluentSpeechCommandsDataset(Dataset):
    actions = ['activate', 'bring', 'change language', 'deactivate', 'decrease', 'increase']
    objs = ['none', 'Chinese', 'English', 'German', 'Korean', 'heat', 'juice', 'lamp', 'lights', 'music', 'newspaper', 'shoes', 'socks', 'volume']
    locations = ['none', 'bedroom', 'kitchen', 'washroom']

    def __init__(self, path, split):
        self.path = path
        self.split = split
        self.data = []

        self._load_data()

    def _load_data(self):
        split_path = self.path / 'data' / f"{self.split.value}_data.csv"
        with open(split_path) as file:
            reader = csv.reader(file)
            reader = itertools.islice(reader, 1, None)
            for row in reader:
                idx, path, _, transcription, action, obj, location = row
                path = self.path / path
                self.data.append(
                    FluentSpeechCommandsInstance(idx, path, transcription, action, obj, location)
                )

    def __getitem__(self, index) -> FluentSpeechCommandsInstance:
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)
