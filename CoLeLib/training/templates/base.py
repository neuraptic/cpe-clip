import warnings
from typing import Optional

import torch
from torch import nn


class BaseTemplate:
    def __init__(
            self,
            seed,
            model: nn.Module,
            device="cpu",
    ):
        seed_everything(seed)
        self.model = model
        self.device = device

        self.is_training = False

        self.experience = None

        self.classes_per_exp = None

        self.num_classes_per_exp = None

        self.num_actual_experience = 0

        self.num_actual_experience_eval = None

    def train(self, experiences):
        self.experiences = experiences
        self.classes_per_exp = experiences.classes_per_exp
        self.num_classes_per_exp = experiences.num_classes_per_exp
        self.is_training = True
        self.model.train()
        self.model.to(self.device)

        if self.experiences.eval_stream is not None:
            assert len(self.experiences.train_stream) == len(self.experiences.eval_stream)
            self.eval_stream = self.experiences.eval_stream
        else:
            self.eval_stream = self.experiences.train_stream

        self._before_training()

        for self.experience in self.experiences.train_stream:
            self._before_training_exp()
            self._train_exp()
            self._after_training_exp()
            # Eval
            self.eval(self.eval_stream[:self.num_actual_experience])

        self._after_training()

    def _before_training(self):
        pass

    def _before_training_exp(self):
        self.num_actual_experience += 1

    def _train_exp(self):
        pass

    def _after_training_exp(self):
        pass

    def _after_training(self):
        pass

    @torch.no_grad()
    def eval(
            self,
            eval_stream
    ):
        self.is_training = False
        self.model.eval()

        self._before_eval()
        for self.experience in eval_stream:
            self._before_eval_exp()
            self._eval_exp()
            self._after_eval_exp()

        self._after_eval()

    def _before_eval(self):
        self.num_actual_experience_eval = 0

    def _before_eval_exp(self):
        self.num_actual_experience_eval += 1

    def _eval_exp(self):
        pass

    def _after_eval_exp(self):
        pass

    def _after_eval(self):
        pass

    
def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    #os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True