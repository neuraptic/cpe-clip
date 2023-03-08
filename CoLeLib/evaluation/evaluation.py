from typing import List
from .metrics import LossMetric, AccuracyMetric, ForgettingMetric
from .loggers import InteractiveLogger, JSONLogger

VALID_METRICS = ['acc', 'loss', 'forgetting']
VALID_LOGGERS = ['interactive', 'json']


class Evaluator:
    def __init__(
            self,
            evaluation_metrics: List[str] = ['loss', 'acc', 'forgetting'],
            loggers: List[str] = ['interactive', 'json'],
            json_file_name: str = 'results.json',
    ):
        for m in evaluation_metrics:
            if m not in VALID_METRICS:
                raise NotImplementedError(f'Evaluation metrics must be among {VALID_METRICS}')
        for l in loggers:
            if l not in VALID_LOGGERS:
                raise NotImplementedError(f'Loggers must be among {VALID_LOGGERS}')

        self.metrics = []
        if 'loss' in evaluation_metrics:
            self.metrics.append(LossMetric())
        if 'acc' in evaluation_metrics:
            self.metrics.append(AccuracyMetric())
        if 'forgetting' in evaluation_metrics:
            if 'acc' not in evaluation_metrics:
                self.metrics.append(AccuracyMetric())
            self.metrics.append(ForgettingMetric())

        self.loggers = []
        if 'interactive' in loggers:
            self.loggers.append(InteractiveLogger())
        if 'json' in loggers:
            self.loggers.append(JSONLogger(json_file_name))

    def update_metrics(self, strategy, status):
        for m in self.metrics:
            m.update(strategy, status)

    def update_loggers(self, strategy, status, metrics):
        for l in self.loggers:
            l.update(strategy, status, metrics)

