from typing import OrderedDict
import torch
import transformers

from segue.modeling_segue import SegueModel
from segue.configuration_segue import SegueConfig
from processing import ASRProcessor

class SegueForAsr(SegueModel):
    config_class = SegueConfig

    def __init__(self, config: SegueConfig) -> None:
        super().__init__(config)
        self.processor = ASRProcessor(
            self.processor.speech_feature_extractor,
            transformers.ByT5Tokenizer.from_pretrained('google/byt5-base'),
            self.processor.conv_sizes,
        )
        self.asr_head = torch.nn.Linear(
            self.speech_encoder.config.hidden_size,
            self.processor.text_tokenizer.vocab_size,
        )
        self.train_backbone = True
    
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def _compute_loss(self, asr_logits, target_ids, input_lengths, target_atn_mask):
        loss = torch.nn.functional.ctc_loss(
            torch.nn.functional.log_softmax(asr_logits, dim=-1).transpose(0, 1),
            target_ids,
            torch.tensor(input_lengths, dtype=torch.int, device=asr_logits.device),
            target_atn_mask.sum(dim=1),
            self.text_encoder.config.pad_token_id,
        )
        return loss

    def forward(self, speech, n_speech_tokens, target_ids=None, target_atn_mask=None, return_logits=False):
        if self.train_backbone:
            outputs = super().forward(speech=speech, n_speech_tokens=n_speech_tokens)
        else:
            with torch.no_grad():
                outputs = super().forward(speech=speech, n_speech_tokens=n_speech_tokens)
        asr_logits = self.asr_head(
            outputs['speech_last_hidden_state']
        )
        predictions = asr_logits.argmax(-1)
        return_ = OrderedDict({
            'predictions': predictions,
        })
        if return_logits:
            return_['logits'] = asr_logits

        if target_ids is not None:
            return_['loss'] = self._compute_loss(asr_logits, target_ids, n_speech_tokens, target_atn_mask)
        
        return return_
