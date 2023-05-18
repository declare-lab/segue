from typing import *
import torch.utils.data
import transformers
import sklearn.linear_model
import numpy as np

import torch

from segue.configuration_segue import SegueConfig
from segue.processing_segue import SegueProcessor

class SegueModel(transformers.PreTrainedModel):
    config_class = SegueConfig

    def __init__(
        self,
        config: SegueConfig,
    ):
        super().__init__(config)
        self.config = config

        self.text_encoder = transformers.AutoModel.from_pretrained(
            config.text_encoder_checkpoint,
        )
        self.speech_encoder = transformers.AutoModel.from_pretrained(
            config.speech_encoder_checkpoint,
        )
        self.text_encoder.requires_grad_(False)
        self._disable_last_dropout()

        tokenizer = transformers.AutoTokenizer.from_pretrained(config.text_encoder_checkpoint)
        feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(
            config.speech_encoder_checkpoint
        )
        self.processor = SegueProcessor(
            feature_extractor,
            tokenizer,
            (self.speech_encoder.config.conv_kernel, self.speech_encoder.config.conv_stride)
        )

        self.train_text_encoder = False

        self.extra_losses_accumulator: MutableMapping[str, List[torch.Tensor]] = {}

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def _disable_last_dropout(self):
        for layer in self.text_encoder.encoder.layer[-1:]:
            layer.output.dropout.p = 0.0

        for layer in self.speech_encoder.encoder.layers[-1:]:
            layer.feed_forward.intermediate_dropout.p = 0.0
            layer.feed_forward.output_dropout.p = 0.0

    def _kd_loss(self, text_pooled_embs: torch.Tensor, speech_pooled_embs: torch.Tensor):
        return torch.nn.functional.mse_loss(
            speech_pooled_embs,
            text_pooled_embs,
        )
    
    def _loss_from_embs(
        self,
        text_embs: torch.Tensor,
        speech_embs: torch.Tensor,
        text_attention_mask: torch.Tensor,
        n_speech_tokens: torch.Tensor,
    ):
        # text_embs = self.text_emb_proj(text_embs)
        loss = self._kd_loss(text_embs, speech_embs)

        return loss

    def forward(self, text = None, speech = None, n_speech_tokens = None, compute_loss: bool = False, output_hidden_states=False):
        return_dict = {}

        if text is not None:
            if self.training:
                self.text_encoder.train(self.train_text_encoder)
            if self.train_text_encoder:
                text_outputs = self.text_encoder(**text)
            else:
                with torch.no_grad():
                    text_outputs = self.text_encoder(**text)
            text_embs = text_outputs.last_hidden_state
            text_pooled_embs = torch.stack([
                sent_embs[atn_mask].mean(dim=0) for sent_embs, atn_mask in zip(text_embs, text['attention_mask'])
            ])
            for k, v in text_outputs.items():
                return_dict['text_' + k] = v
            return_dict['text_pooled_embs'] = text_pooled_embs
        
        if speech is not None:
            speech_outputs = self.speech_encoder(**speech, output_hidden_states=output_hidden_states)
            speech_embs = speech_outputs.last_hidden_state
            speech_pooled_embs = torch.stack([
                sent_embs[:n_tokens].mean(dim=0) for sent_embs, n_tokens in zip(speech_embs, n_speech_tokens)
            ])
            for k, v in speech_outputs.items():
                return_dict['speech_' + k] = v
            return_dict['speech_pooled_embs'] = speech_pooled_embs

        if compute_loss:
            assert text is not None and speech is not None
            return_dict['loss'] = self._loss_from_embs(
                text_pooled_embs, speech_pooled_embs,
                text['attention_mask'], n_speech_tokens, 
            )

        return return_dict
    
    def _accumulate_extra_metric(self, name, value):
        buffer = self.extra_losses_accumulator.get(name, [])
        buffer.append(value.detach().cpu())
        self.extra_losses_accumulator[name] = buffer
    
    def pop_extra_log_metrics(self):
        metrics = {
            k: torch.tensor(v).mean().item() for k, v in self.extra_losses_accumulator.items()
        }
        self.extra_losses_accumulator.clear()
        return metrics

class SegueForRegression(SegueModel):
    config_class = SegueConfig

    def __init__(self, config: SegueConfig) -> None:
        super().__init__(config)
        self.tokenizer = self.processor.text_tokenizer
        self.regression_head = torch.nn.Sequential(
            torch.nn.Linear(
                self.speech_encoder.config.hidden_size,
                1,
            )
        )
        # set a dummy value first then use the setter
        self._train_backbone = None
        self.train_backbone = config.train_backbone
    
    @property
    def train_backbone(self):
        return self._train_backbone
    @train_backbone.setter
    def train_backbone(self, value: bool):
        self.speech_encoder.requires_grad_(value)
        self._train_backbone = value

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=torch.sqrt(2/torch.tensor(768)))
            if module.bias is not None:
                module.bias.data.zero_()
    
    def _compute_loss(self, predictions, labels):
        return torch.nn.functional.mse_loss(predictions, labels)
    
    def forward(self, speech=None, text=None, n_speech_tokens = None, labels = None):
        if self.training:
            self.speech_encoder.train(self.train_backbone)
        outputs = super().forward(speech=speech, n_speech_tokens=n_speech_tokens, text=text)
        speech_embs = outputs.get('speech_pooled_embs', None)
        text_embs = outputs.get('text_pooled_embs', None)
        if speech_embs is not None and text_embs is not None:
            logits = (speech_embs + text_embs) / 2
        else:
            logits = speech_embs if speech_embs is not None else text_embs
        predictions = self.regression_head(
            logits
        ).squeeze(-1)
        return_ = {
            'predictions': predictions,
        }

        if labels is not None:
            return_['loss'] = self._compute_loss(predictions, labels)
        
        return return_

class SegueForClassification(SegueModel):
    config_class = SegueConfig

    def __init__(self, config: SegueConfig) -> None:
        super().__init__(config)
        self.is_multilabel = isinstance(config.n_classes, Sequence)
        self.tokenizer = self.processor.text_tokenizer
        self.classification_heads = torch.nn.ModuleList([
            torch.nn.Linear(
                self.speech_encoder.config.hidden_size,
                label_n_classes,
            )
            for label_n_classes in (config.n_classes if self.is_multilabel else [config.n_classes])
        ])
        # set a dummy value first then use the setter
        self._train_backbone = None
        self.train_backbone = config.train_backbone
        self.ce_loss = torch.nn.CrossEntropyLoss()
    
    @property
    def train_backbone(self):
        return self._train_backbone
    @train_backbone.setter
    def train_backbone(self, value: bool):
        self.speech_encoder.requires_grad_(value)
        self._train_backbone = value

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=torch.sqrt(2/torch.tensor(768)))
            if module.bias is not None:
                module.bias.data.zero_()
    
    def _compute_loss(self, predictions, labels):
        return self.ce_loss(predictions, labels)
    
    def forward(self, speech=None, text=None, n_speech_tokens = None, labels = None):
        if not self.is_multilabel:
            labels = labels.unsqueeze(-1)
        if self.training:
            self.speech_encoder.train(self.train_backbone)
        outputs = super().forward(speech=speech, n_speech_tokens=n_speech_tokens, text=text)
        speech_embs = outputs.get('speech_pooled_embs', None)
        text_embs = outputs.get('text_pooled_embs', None)
        if speech_embs is not None and text_embs is not None:
            logits = (speech_embs + text_embs) / 2
        else:
            logits = speech_embs if speech_embs is not None else text_embs
        predictions = [head(logits) for head in self.classification_heads]

        return_ = {
            'predictions': predictions if self.is_multilabel else predictions[0],
        }

        if labels is not None:
            loss = torch.tensor(0.)
            for p, l in zip(predictions, labels.mT):
                loss = loss + self._compute_loss(p, l)
            return_['loss'] = loss

        return return_

class SegueForAnalyticRegression(torch.nn.Module):
    """
    A SEGUE model with scikit-learn's ridge regression on top.
    Meant as a wrapper to use the functionalities of `trainsformers.Trainer`.
    Not meant for iterative training - iterating through a sample merely computes
    an embedding and adds it to the list of seen data. Make sure to set number of
    epochs = 1 to make sure each data point is seen exactly once.
    """

    def __init__(self, segue: SegueModel, alpha=.5) -> None:
        super().__init__()
        self.segue = segue
        self.segue.eval()
        self.linear_regression = sklearn.linear_model.Ridge(alpha=alpha, solver='svd')
        self.data = []
    
    def train(self, train=True):
        super().train(train)
        if train:
            self.segue.eval()
    
    def fit(self):
        X, y = zip(*self.data)
        X = np.concatenate(X)
        y = np.concatenate(y)
        self.linear_regression.fit(X, y)
    
    def forward(self, speech=None, text=None, n_speech_tokens = None, labels = None):
        with torch.no_grad():
            outputs = self.segue(speech=speech, n_speech_tokens=n_speech_tokens, text=text)
        speech_embs = outputs.get('speech_pooled_embs', None)
        text_embs = outputs.get('text_pooled_embs', None)
        if speech_embs is not None and text_embs is not None:
            logits = torch.cat((speech_embs, text_embs), dim=-1)
        else:
            logits = speech_embs if speech_embs is not None else text_embs
        if self.training:
            self.data.append(
                (logits.cpu().numpy(), labels.cpu().numpy())
            )
            return {
                'predictions': torch.zeros(labels.shape),
                'loss': torch.tensor(torch.nan, requires_grad=True),
            }
        else:
            predictions = self.linear_regression.predict(logits.cpu().numpy())
            return {
                'predictions': torch.from_numpy(predictions),
                'loss': torch.tensor(torch.nan, requires_grad=True),
            }
