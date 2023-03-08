import warnings
from typing import Optional, List

import math
import numpy as np

import torch
from torch import nn

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from transformers import CLIPProcessor

from CoLeLib.training.templates import SupervisedTemplate
from CoLeLib.models import CLIPParameterEfficient

class CLIPPE(SupervisedTemplate):
    def __init__(self,
            seed: int,
            L_g: int, 
            deep_g: int, 
            text_deep_replace_method: str,
            vision_deep_replace_method: str,
            regularization_method: str = 'balance',
            manual_prompt: str = '[].',
            lr: float = 0.00325,
            gradient_accumulation_steps: int = 1,
            grad_clip_max_norm = None,
            train_mb_size_base_class = 4,
            train_epochs_base_class = 3,
            use_scheduler = True,
            train_mb_size: int = 4,
            eval_mb_size: int = 4,
            train_epochs: int = 5,
            device='cpu',
            evaluation_metrics: List[str] = ['acc', 'loss', 'forgetting'],
            loggers: List[str] = ['interactive', 'json'],
            json_file_name: str = 'results.json',
    ):
        
        model = CLIPParameterEfficient(
            L_g = L_g, 
            deep_g = deep_g, 
            text_deep_replace_method = text_deep_replace_method,
            vision_deep_replace_method = vision_deep_replace_method
        )
            
        self.regularization_method = regularization_method

        self.lr = lr
        self.use_scheduler = use_scheduler
        self.train_mb_size_base_class = train_mb_size_base_class
        self.train_epochs_base_class = train_epochs_base_class

        super().__init__(
            model=model,
            optimizer=None,
            criterion=nn.CrossEntropyLoss(),
            gradient_accumulation_steps=gradient_accumulation_steps,
            grad_clip_max_norm=grad_clip_max_norm,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            seed=seed,
            evaluation_metrics=evaluation_metrics,
            loggers=loggers,
            json_file_name=json_file_name,
        )
                
        self.actual_text_labels = None
        self.text_tokens = None
        self.attn_mask = None

        self.manual_prompt = manual_prompt
        self.prompt_labels = []
        self.prompt_labels = [self.manual_prompt.replace('[]', i) for i in self.prompt_labels]
        
        self.text_preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16").tokenizer

    def forward(self):
        logits = self.model(self.mb_x, self.text_tokens, self.attn_mask)
        return logits

    def criterion(self):
        loss = self._criterion(self.mb_output, self.mb_y)
        return loss

    def make_optimizer(self):
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                          lr=self.lr,
                                          momentum=0.9,
                                          weight_decay=1e-5)
        if self.use_scheduler:
            self.scheduler = CosineAnnealingWarmupRestarts(
                self.optimizer,
                first_cycle_steps=200,
                cycle_mult=1.0,
                max_lr=0.1, min_lr=0.001,
                warmup_steps=50, gamma=1.0
            )

    def model_adaptation(self, model=None):
        if model is None:
            model = self.model
        if self.is_training:
            if self.num_actual_experience > 1:
                model = super().model_adaptation(model)

        return model.to(self.device)

    def _after_forward(self):
        old_nclasses = sum(self.num_classes_per_exp[:self.num_actual_experience-1])
        self.mb_output[:, :old_nclasses] = -9999
        super()._after_forward()

    def _after_training_epoch(self):
        if self.use_scheduler:
            self.scheduler.step()
        super()._after_training_epoch()
     
    def _before_training_exp(self):
        if self.num_actual_experience == 0:
            self.train_mb_size = self.train_mb_size_base_class
            self.train_epochs = self.train_epochs_base_class
        else:
            self.train_mb_size = 4
            self.train_epochs = 5
        
        if self.regularization_method == 'freeze':
            if self.num_actual_experience > 1:
                self.model.g_values.requires_grad = False
                self.model.prompt_proj.requires_grad = False
                
        super()._before_training_exp()
        self.actual_text_labels = [self.experiences.text_label_mapping[i] for i in self.classes_per_exp[self.num_actual_experience-1]]
        self.prompt_labels += [self.manual_prompt.replace('[]', i) for i in self.actual_text_labels]
        out_text_tokens = self.text_preprocess(self.prompt_labels, padding=True, return_tensors="pt")
        self.text_tokens = out_text_tokens["input_ids"]
        self.attn_mask = out_text_tokens["attention_mask"]
            
        self.text_tokens = self.text_tokens.to(self.device)
        self.attn_mask = self.attn_mask.to(self.device)
        
    def _before_update(self):
        if self.num_actual_experience > 1:
            if self.regularization_method == 'balance':
                reg_lambda = self.num_classes_per_exp[self.num_actual_experience-1] / sum(self.num_classes_per_exp[:self.num_actual_experience])
                self.model.g_values.grad *= reg_lambda
                self.model.prompt_proj.weight.grad *= reg_lambda
                self.model.prompt_proj.bias.grad *= reg_lambda