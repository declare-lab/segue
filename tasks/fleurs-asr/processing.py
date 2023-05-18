import itertools
from typing import *
import numpy as np
from transformers import ProcessorMixin, PreTrainedTokenizer, SequenceFeatureExtractor
import transformers

class ASRProcessor(ProcessorMixin):
    feature_extractor_class = 'Wav2Vec2FeatureExtractor'
    tokenizer_class = 'ByT5Tokenizer'

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

    def __call__(self, text, audio, sampling_rate: int):
        text_input = self.text_tokenizer(text)
        speech_input = self.speech_feature_extractor(audio, sampling_rate=sampling_rate)
        speech_input = {k:v[0] for k, v in speech_input.items()}
        n_samples: int
        if 'attention_mask' in speech_input:
            n_samples = speech_input['attention_mask'].sum()
        else:
            n_samples = np.not_equal(speech_input['input_values'], 0).sum()
        n_speech_tokens = self.compute_n_tokens(n_samples)

        return {
            'target_ids': text_input['input_ids'],
            'target_atn_mask': text_input['attention_mask'],
            'speech': speech_input,
            'n_speech_tokens': [n_speech_tokens],
        }
    
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
        text_input = self.text_tokenizer.pad(
            [
                {
                    'input_ids': i['target_ids'],
                    'attention_mask': i['target_atn_mask'],
                }
                for i in encoded_inputs
            ],
            padding,
            max_length,
            pad_to_multiple_of,
            return_attention_mask,
            return_tensors,
        )
        speech_input = self.speech_feature_extractor.pad(
            [i['speech'] for i in encoded_inputs],
            padding,
            max_length,
            pad_to_multiple_of = pad_to_multiple_of,
            return_attention_mask = return_attention_mask,
            return_tensors = return_tensors,
        )
        n_speech_tokens = list(itertools.chain.from_iterable(
            i['n_speech_tokens'] for i in encoded_inputs
        ))
        return_dict = {
            'target_ids': text_input['input_ids'],
            'target_atn_mask': text_input['attention_mask'],
            'speech': speech_input,
            'n_speech_tokens': n_speech_tokens,
        }
        if any('compute_loss' in i and i['compute_loss'] for i in encoded_inputs):
            return_dict['compute_loss'] = True
        return return_dict
