from enum import Enum
from os import PathLike
from pathlib import Path
import numpy as np
from torch.utils.data import IterableDataset
from dataclasses import dataclass
from mmsdk import mmdatasdk as md
from typing import *
import regex as re
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import EvalPrediction
from sys import stderr
import soundfile


T = TypeVar('T')

class DataSplitType(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'

@dataclass
class BySplit(Generic[T]):
    train: T
    val: T
    test: T

    def __getitem__(self, key: DataSplitType) -> T:
        if key == DataSplitType.TRAIN:
            return self.train
        elif key == DataSplitType.VAL:
            return self.val
        elif key == DataSplitType.TEST:
            return self.test
        else:
            raise TypeError("`key` should be a value of the enum type DataSplitType")

@dataclass(order=True)
class MoseiSegment:
    name: str
    label: float
    text: str

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

class MoseiDataset(IterableDataset):
    def __init__(
        self,
        mosei_path: Union[str, PathLike],
        preprocessed_path: Union[str, PathLike],
        split: DataSplitType,
    ):
        self.mosei_path = Path(mosei_path)
        self.preprocessed_path = Path(preprocessed_path)
        self.split = split
        self.data: List[MoseiSegment]

        preprocessed_file = self.preprocessed_path / f"{split.value}.pt"
        if not preprocessed_file.is_file():
            self.data = self._preprocess()[split]
        else:
            self.data = torch.load(preprocessed_file)

    def __getitem__(self, index: int) -> MoseiSegment:
        return self.data.__getitem__(index)

    def __len__(self):
        return self.data.__len__()
    
    def __iter__(self):
        return self.data.__iter__()

    def _preprocess(self) -> BySplit[List[MoseiSegment]]:
        MoseiDataset._try_download_dataset(self.mosei_path)

        # load data from mmsdk
        text_field = 'CMU_MOSEI_TimestampedWords'
        features = (text_field,)
        recipe = {feat: (self.mosei_path / f"{feat}.csd").as_posix() for feat in features}
        dataset = md.mmdataset(recipe)

        label_field = 'CMU_MOSEI_Labels'
        recipe = {label_field: (self.mosei_path / f"{label_field}.csd").as_posix()}
        # dataset = md.mmdataset(recipe)
        dataset.add_computational_sequences(recipe, destination=None)
        dataset.align(label_field)

        # preprocess data
        vid_ids_by_split = BySplit(
            md.cmu_mosei.standard_folds.standard_train_fold,
            md.cmu_mosei.standard_folds.standard_valid_fold,
            md.cmu_mosei.standard_folds.standard_test_fold,
        )

        vid_id_capture_pattern = re.compile('(.*)\[(.*)\]')

        data_by_split: BySplit[List[MoseiSegment]] = BySplit([], [], [])
        missing_videos = []
        for segment_name in dataset[label_field].keys():
            match = re.search(vid_id_capture_pattern, segment_name)
            vid_id = match.group(1)
            seg_idx = int(match.group(2))
            text = ' '.join(
                word.item().decode()
                for word in dataset[text_field][segment_name]['features']
                if word.item() != b'sp'
            )
            label: np.ndarray = dataset[label_field][segment_name]['features'][0,:1]

            for split in DataSplitType:
                if vid_id in vid_ids_by_split[split]:
                    raw_path = self.mosei_path / 'Raw' / 'Audio' / 'FLAC' / f"{vid_id}_{seg_idx}.flac"
                    if raw_path.is_file():
                        data_by_split[split].append(
                            MoseiSegment(
                                segment_name,
                                torch.from_numpy(np.nan_to_num(label)).item(),
                                text,
                                raw_path,
                            )
                        )
                    else:
                        print(f"Video {raw_path.name} not found.", file=stderr)
                        missing_videos.append(raw_path.name)
                    break
            else:
                print(f"Video {vid_id} does not belong to any split.")

        for split in DataSplitType:
            self.preprocessed_path.mkdir(parents=True, exist_ok=True)
            torch.save(data_by_split[split], self.preprocessed_path / f"{split.value}.pt")

        if len(missing_videos) > 0:
            with open(self.preprocessed_path / 'missing_segments.txt', 'w') as file:
                for name in missing_videos:
                    print(name, file=file)
            print("Missing videos:", len(missing_videos))

        return data_by_split

    @staticmethod
    def _try_download_dataset(mosei_path: Path):
        if not mosei_path.is_dir():
            mosei_path.mkdir(parents=True)

        dataset = md.cmu_mosei

        try:
            md.mmdataset(dataset.raw, mosei_path.as_posix())
        except RuntimeError:
            print("Raw data have been downloaded previously.")

        try:
            md.mmdataset(dataset.labels, mosei_path.as_posix())
        except RuntimeError:
            print("Labels have been downloaded previously.")


def compute_metrics(p: EvalPrediction):
    preds = np.squeeze(p.predictions)
    truth = np.array(p.label_ids[0]).squeeze()
    non_zeros = np.array([i for i, e in enumerate(truth) if e != 0])
    mae = np.mean(np.absolute(preds - truth))
    corr = np.corrcoef(preds, truth)[0][1]
    f_score = f1_score((truth[non_zeros] > 0), (preds[non_zeros] > 0), average='weighted')
    # pos - neg
    binary_truth = (truth[non_zeros] >= 0)
    binary_preds = (preds[non_zeros] >= 0)
    acc2 = accuracy_score(binary_truth, binary_preds)

    truth7 = truth.round()
    preds7 = preds.round()
    acc7 = accuracy_score(truth7, preds7)

    return {
        "mae": mae,
        "corr": corr,
        "f1" : f_score,
        "accuracy": acc2,
        "acc7": acc7
    }
