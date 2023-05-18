from transformers import PretrainedConfig
from typing import *

class SegueConfig(PretrainedConfig):
    model_type = 'segue'

    def __init__(
        self,
        text_encoder_checkpoint: str = 'sentence-transformers/all-mpnet-base-v2',
        speech_encoder_checkpoint: str = 'facebook/wav2vec2-base-960h',
        train_backbone: bool = True,
        n_classes: Union[int, Sequence[int]] = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.text_encoder_checkpoint = text_encoder_checkpoint
        self.speech_encoder_checkpoint = speech_encoder_checkpoint
        self.train_backbone = train_backbone
        self.n_classes = n_classes
