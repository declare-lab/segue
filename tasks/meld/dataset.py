import csv
from enum import Enum
import itertools
from os import PathLike
from pathlib import Path
import numpy as np
from torch.utils.data import IterableDataset
from dataclasses import dataclass
from typing import *
import soundfile


T = TypeVar('T')

class DataSplitType(Enum):
    TRAIN = 'train'
    DEV = 'dev'
    TEST = 'test'

class Emotion(Enum):
    NEUTRAL = 0
    JOY = 1
    SADNESS = 2
    ANGER = 3
    SURPRISE = 4
    FEAR = 5
    DISGUST = 6

@dataclass(order=True)
class MeldInstance:
    name: str
    text: str
    sentiment: int
    emotion: Emotion

    _path: Path
    _audio: Optional[np.ndarray] = None
    _sampling_rate: Optional[int] = None

    def _ensure_audio_loaded(self):
        if self._audio is None or self._sampling_rate is None:
            self._audio, self._sampling_rate = soundfile.read(self._path)
    
    @property
    def audio(self):
        self._ensure_audio_loaded()
        return self._audio

    @property
    def sampling_rate(self):
        self._ensure_audio_loaded()
        return self._sampling_rate

class MeldDataset(IterableDataset):
    def __init__(
        self,
        meld_path: Union[str, PathLike],
        split: DataSplitType,
    ):
        self.meld_path = Path(meld_path)
        self.split = split
        self.data: MutableSequence[MeldInstance] = []

        self._load_data()

    def _load_data(self):
        split_path = self.meld_path / f"{self.split.value}_sent_emo.csv"
        with open(split_path) as file:
            reader = csv.reader(file)
            reader = itertools.islice(reader, 1, None)
            for row in reader:
                sr_no, utterance, _, emotion, sentiment, dia_id, utt_id, _, _, _, _ = row
                sentiment = 1 if sentiment == 'positive' else 0 if sentiment == 'neutral' else -1
                emotion = [e for e in Emotion if e.name.lower() == emotion][0]
                path = self.meld_path / self.split.value / f"dia{dia_id}_utt{utt_id}.flac"
                if path.is_file():
                    self.data.append(
                        MeldInstance(sr_no, utterance, sentiment, emotion, path)
                    )

    def __getitem__(self, index: int) -> MeldInstance:
        return self.data.__getitem__(index)

    def __len__(self):
        return self.data.__len__()
    
    def __iter__(self):
        return self.data.__iter__()
