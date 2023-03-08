import sys
from tqdm import tqdm

import json


class InteractiveLogger:
    def __init__(self):
        self._pbar = None

    @property
    def _progress(self):
        if self._pbar is None:
            self._pbar = tqdm(leave=True, position=0, file=sys.stdout)
        return self._pbar

    def _end_progress(self):
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

    def update(self, strategy, status, metrics):
        if status == 'before_training_exp':
            print(f'Start of experience {strategy.num_actual_experience}')
            print(f'Current classes: {strategy.classes_per_exp[strategy.num_actual_experience-1]}')

        elif status == 'train_exp':
            print(f'-- >> Start of training phase << --')

        elif status == 'after_training_epoch':
            self._end_progress()
            print(f'Epoch {strategy.epoch} ended')
            for m in metrics:
                key = f'{m.name}/Epoch{strategy.epoch}/Train/Exp{strategy.num_actual_experience}'
                if m.metric.get(key, None) is not None:
                    print(' '*10 + f'{key} = {round(m.metric[key],4)}')

        elif status == 'after_training_exp':
            print(f'-- >> End of training phase << --')

        elif status == 'before_eval':
            print(f'-- >> Start of eval phase << --')

        elif status == 'before_eval_exp':
            print(f'-- Starting eval on experience {strategy.num_actual_experience_eval} --')
            self._progress.total = len(strategy.dataloader)

        elif status == 'after_eval_exp':
            self._end_progress()
            print(f'-- Eval on experience {strategy.num_actual_experience_eval} ended --')
            for m in metrics:
                key = f'{m.name}/Eval/Exp{strategy.num_actual_experience}/Exp{strategy.num_actual_experience_eval}'
                if m.metric.get(key, None) is not None:
                    print(' '*10 + f'{key} = {round(m.metric[key],4)}')

        elif status == 'after_eval':
            print(f'-- >> End of eval phase << --')
            for m in metrics:
                key = f'{m.name}/Eval/Exp{strategy.num_actual_experience}'
                if m.metric.get(key, None) is not None:
                    print(' '*10 + f'{key} = {round(m.metric[key],4)}')

            print('-'*10 + f' END OF EXPERIENCE {strategy.num_actual_experience} ' + '-'*10)

        elif status == 'after_training':
            print('----- AVERAGE METRICS -----')
            for m in metrics:
                key = f'Avg{m.name}'
                if m.metric.get(key, None) is not None:
                    print(' '*10 + f'{key} = {round(m.metric[key],4)}')

        elif status == 'after_training_iteration':
            self._pbar.update()
            self._pbar.refresh()

        elif status == 'after_eval_iteration':
            self._pbar.update()
            self._pbar.refresh()

        elif status == 'before_training_epoch':
            self._progress.total = len(strategy.dataloader)

        else:
            pass


class JSONLogger:
    def __init__(self, file_name):
        self.file_name = file_name

    def update(self, strategy, status, metrics):
        if status == 'after_training':
            json_to_save = {}
            for m in metrics:
                json_to_save[m.name] = m.metric

            with open(self.file_name, 'w') as f:
                json.dump(json_to_save, f)

