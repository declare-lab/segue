from typing import Dict
from transformers import Trainer

class CustomTrainer(Trainer):
    """
    A custom trainer that logs extra custom metrics in tensorboard.
    Expects a `pop_extra_log_metrics()` method in `self.model`.
    """
    def log(self, data):
        extra_metrics = self.model.pop_extra_log_metrics()
        if 'eval_loss' in data:
            for k, v in extra_metrics.items():
                data['eval_' + k] = v
        else:
            data.update(extra_metrics)
        super().log(data)
