import itertools
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from transformers import ProcessorMixin, PreTrainedTokenizer, SequenceFeatureExtractor
import transformers

class SegueProcessor(ProcessorMixin):
    feature_extractor_class = 'Wav2Vec2FeatureExtractor'
    tokenizer_class = 'MPNetTokenizerFast'

    def __init__(
        self,
        speech_feature_extractor,
        text_tokenizer,
        conv_sizes: Tuple[List[int], List[int]],
    ):
        super().__init__(
            speech_feature_extractor,
            text_tokenizer,
        )

        self.text_tokenizer: PreTrainedTokenizer = text_tokenizer
        self.speech_feature_extractor: SequenceFeatureExtractor = \
            speech_feature_extractor
        self.conv_sizes = conv_sizes

    def __call__(self, text = None, audio = None, sampling_rate: Optional[int] = None):
        return_ = {}
        if text is not None:
            text_input = self.text_tokenizer(text)
            return_['text'] = text_input
        if audio is not None:
            speech_input = self.speech_feature_extractor(audio, sampling_rate=sampling_rate)
            speech_input = {k:v[0] for k, v in speech_input.items()}
            n_samples: int
            if 'attention_mask' in speech_input:
                n_samples = speech_input['attention_mask'].sum()
            else:
                n_samples = np.not_equal(speech_input['input_values'], 0).sum()
            n_speech_tokens = self.compute_n_tokens(n_samples)
            return_['speech'] = speech_input
            return_['n_speech_tokens'] = [n_speech_tokens]

        return return_
    
    def compute_n_tokens(self, n_samples: int) -> int:
        n = n_samples
        for kernel_size, stride in zip(*self.conv_sizes):
            n = (n - (kernel_size - stride)) // stride
        return n

    def pad(
        self,
        encoded_inputs,
        padding: Union[bool, str, transformers.utils.generic.PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Union[str, transformers.utils.generic.TensorType, None] = None,
    ):
        return_dict = {
        }
        if all('text' in i for i in encoded_inputs):
            text_input = self.text_tokenizer.pad(
                [i['text'] for i in encoded_inputs],
                padding,
                max_length,
                pad_to_multiple_of,
                return_attention_mask,
                return_tensors,
            )
            return_dict['text'] = text_input
        if all('speech' in i for i in encoded_inputs):
            speech_input = self.speech_feature_extractor.pad(
                [i['speech'] for i in encoded_inputs],
                padding,
                max_length,
                pad_to_multiple_of = pad_to_multiple_of,
                return_attention_mask = return_attention_mask,
                return_tensors = return_tensors,
            )
            return_dict['speech'] = speech_input
        if all('n_speech_tokens' in i for i in encoded_inputs):
            n_speech_tokens = torch.tensor(list(itertools.chain.from_iterable(
                i['n_speech_tokens'] for i in encoded_inputs
            )))
            return_dict['n_speech_tokens'] = n_speech_tokens
        if any('compute_loss' in i and i['compute_loss'] for i in encoded_inputs):
            return_dict['compute_loss'] = True
        if all('labels' in i for i in encoded_inputs):
            return_dict['labels'] = torch.tensor(list(itertools.chain.from_iterable(
                i['labels'] for i in encoded_inputs
            )))
        return return_dict
