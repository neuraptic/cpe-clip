import torch
import numpy as np

class LossMetric:
    def __init__(self):
        self.name = 'Loss'
        self.metric = {}
        self.counter = {}

    def result(self, strategy):
        return strategy.loss.item()

    def update(self, strategy, status):
        key = 'Loss'
        if status == 'after_training_iteration':
            key += f'/Epoch{strategy.epoch}/Train/Exp{strategy.num_actual_experience}'
            if key in self.metric:
                self.metric[key] += self.result(strategy)*len(strategy.mb_output)
                self.counter[key] += len(strategy.mb_output)
            else:
                self.metric[key] = self.result(strategy)*len(strategy.mb_output)
                self.counter[key] = len(strategy.mb_output)

        elif status == 'after_eval_iteration':
            key += f'/Eval/Exp{strategy.num_actual_experience}/Exp{strategy.num_actual_experience_eval}'
            if key in self.metric:
                self.metric[key] += self.result(strategy)*len(strategy.mb_output)
                self.counter[key] += len(strategy.mb_output)
            else:
                self.metric[key] = self.result(strategy)*len(strategy.mb_output)
                self.counter[key] = len(strategy.mb_output)

        elif status == 'after_training_epoch':
            self.metric[f'Loss/Epoch{strategy.epoch}/Train/Exp{strategy.num_actual_experience}'] /= self.counter[f'Loss/Epoch{strategy.epoch}/Train/Exp{strategy.num_actual_experience}']

        elif status == 'after_eval_exp':
            self.metric[f'Loss/Eval/Exp{strategy.num_actual_experience}/Exp{strategy.num_actual_experience_eval}'] /= self.counter[f'Loss/Eval/Exp{strategy.num_actual_experience}/Exp{strategy.num_actual_experience_eval}']

        elif status == 'after_eval':
            self.metric[f'Loss/Eval/Exp{strategy.num_actual_experience}'] = 0

            for i in range(strategy.num_actual_experience):
                self.metric[f'Loss/Eval/Exp{strategy.num_actual_experience}'] += self.metric[f'Loss/Eval/Exp{strategy.num_actual_experience}/Exp{i+1}']

            self.metric[f'Loss/Eval/Exp{strategy.num_actual_experience}'] /= strategy.num_actual_experience

        else:
            pass


#class AccuracyMetric:
#    def __init__(self):
#        self.name = 'Acc'
#        self.metric = {}
#        self.counter = {}
#
#    def result(self, strategy):
#        return float(torch.sum(torch.eq(torch.argmax(strategy.mb_output, dim=1), strategy.mb_y)))
#
#    def update(self, strategy, status):
#        key = 'Acc'
#        if status == 'after_training_iteration':
#            key += f'/Epoch{strategy.epoch}/Train/Exp{strategy.num_actual_experience}'
#            if key in self.metric:
#                self.metric[key] += self.result(strategy)
#                self.counter[key] += len(strategy.mb_output)
#            else:
#                self.metric[key] = self.result(strategy)
#                self.counter[key] = len(strategy.mb_output)
#
#        elif status == 'after_eval_iteration':
#            key += f'/Eval/Exp{strategy.num_actual_experience}/Exp{strategy.num_actual_experience_eval}'
#            if key in self.metric:
#                self.metric[key] += self.result(strategy)
#                self.counter[key] += len(strategy.mb_output)
#            else:
#                self.metric[key] = self.result(strategy)
#                self.counter[key] = len(strategy.mb_output)
#
#        elif status == 'after_training_epoch':
#            self.metric[f'Acc/Epoch{strategy.epoch}/Train/Exp{strategy.num_actual_experience}'] /= self.counter[f'Acc/Epoch{strategy.epoch}/Train/Exp{strategy.num_actual_experience}']
#
#        elif status == 'after_eval_exp':
#            self.metric[f'Acc/Eval/Exp{strategy.num_actual_experience}/Exp{strategy.num_actual_experience_eval}'] /= self.counter[f'Acc/Eval/Exp{strategy.num_actual_experience}/Exp{strategy.num_actual_experience_eval}']
#
#        elif status == 'after_eval':
#            self.metric[f'Acc/Eval/Exp{strategy.num_actual_experience}'] = 0
#
#            for i in range(strategy.num_actual_experience):
#                self.metric[f'Acc/Eval/Exp{strategy.num_actual_experience}'] += self.metric[f'Acc/Eval/Exp{strategy.num_actual_experience}/Exp{i+1}']
#
#            self.metric[f'Acc/Eval/Exp{strategy.num_actual_experience}'] /= strategy.num_actual_experience
#
#        elif status == 'after_training':
#            self.metric['AvgAcc'] = 0
#            for i in range(strategy.num_actual_experience):
#                for j in range(i+1):
#                    self.metric['AvgAcc'] += self.metric[f'Acc/Eval/Exp{i+1}/Exp{j+1}']
#            self.metric['AvgAcc'] /= (strategy.num_actual_experience*(strategy.num_actual_experience+1) / 2)
#
#        else:
#            pass


class AccuracyMetric:
    def __init__(self):
        self.name = 'Acc'
        self.metric = {}
        self.counter = {}

    def result(self, strategy):
        return float(torch.sum(torch.eq(torch.argmax(strategy.mb_output, dim=1), strategy.mb_y)))

    def update(self, strategy, status):
        key = 'Acc'
        if status == 'after_training_iteration':
            key += f'/Epoch{strategy.epoch}/Train/Exp{strategy.num_actual_experience}'
            if key in self.metric:
                self.metric[key] += self.result(strategy)
                self.counter[key] += len(strategy.mb_output)
            else:
                self.metric[key] = self.result(strategy)
                self.counter[key] = len(strategy.mb_output)

        elif status == 'after_eval_iteration':
            key += f'/Eval/Exp{strategy.num_actual_experience}/Exp{strategy.num_actual_experience_eval}'
            if key in self.metric:
                self.metric[key] += self.result(strategy)
                self.counter[key] += len(strategy.mb_output)
            else:
                self.metric[key] = self.result(strategy)
                self.counter[key] = len(strategy.mb_output)

        elif status == 'after_training_epoch':
            self.metric[f'Acc/Epoch{strategy.epoch}/Train/Exp{strategy.num_actual_experience}'] /= self.counter[f'Acc/Epoch{strategy.epoch}/Train/Exp{strategy.num_actual_experience}']

        elif status == 'after_eval_exp':
            self.metric[f'Acc/Eval/Exp{strategy.num_actual_experience}/Exp{strategy.num_actual_experience_eval}'] /= self.counter[f'Acc/Eval/Exp{strategy.num_actual_experience}/Exp{strategy.num_actual_experience_eval}']

        elif status == 'after_eval':
            self.metric[f'Acc/Eval/Exp{strategy.num_actual_experience}'] = 0
            self.counter[f'Acc/Eval/Exp{strategy.num_actual_experience}'] = 0

            for i in range(strategy.num_actual_experience):
                self.metric[f'Acc/Eval/Exp{strategy.num_actual_experience}'] += self.metric[f'Acc/Eval/Exp{strategy.num_actual_experience}/Exp{i+1}'] * self.counter[f'Acc/Eval/Exp{strategy.num_actual_experience}/Exp{i + 1}']
                self.counter[f'Acc/Eval/Exp{strategy.num_actual_experience}'] += self.counter[f'Acc/Eval/Exp{strategy.num_actual_experience}/Exp{i + 1}']

            self.metric[f'Acc/Eval/Exp{strategy.num_actual_experience}'] /= self.counter[f'Acc/Eval/Exp{strategy.num_actual_experience}']

        elif status == 'after_training':
            self.metric['AvgAcc'] = 0
            self.counter['AvgAcc'] = 0
            for i in range(strategy.num_actual_experience):
                for j in range(i+1):
                    self.metric['AvgAcc'] += self.metric[f'Acc/Eval/Exp{i+1}/Exp{j+1}'] * self.counter[f'Acc/Eval/Exp{i + 1}/Exp{j + 1}']
                    self.counter['AvgAcc'] += self.counter[f'Acc/Eval/Exp{i + 1}/Exp{j + 1}']
            self.metric['AvgAcc'] /= self.counter['AvgAcc']#(strategy.num_actual_experience*(strategy.num_actual_experience+1)/2)

        else:
            pass


class ForgettingMetric:
    def __init__(self):
        self.name = 'Forgetting'
        self.metric = {}

    def result(self, acc_results, acc_actual):
        return np.max(np.array(acc_results) - acc_actual)

    def update(self, strategy, status):
        if strategy.num_actual_experience > 1:
            acc_results = None
            for m in strategy.evaluator.metrics:
                if m.name == 'Acc':
                    acc_results = m.metric
            if status == 'after_eval_exp':
                if strategy.num_actual_experience_eval < strategy.num_actual_experience:
                    accs = []
                    for i in range(strategy.num_actual_experience_eval, strategy.num_actual_experience):
                        accs.append(acc_results[f'Acc/Eval/Exp{i}/Exp{strategy.num_actual_experience_eval}'])
                    acc_actual = acc_results[f'Acc/Eval/Exp{strategy.num_actual_experience}/Exp{strategy.num_actual_experience_eval}']
                    self.metric[f'Forgetting/Eval/Exp{strategy.num_actual_experience}/Exp{strategy.num_actual_experience_eval}'] = self.result(accs, acc_actual)

            elif status == 'after_eval':
                self.metric[f'Forgetting/Eval/Exp{strategy.num_actual_experience}'] = 0
                for i in range(strategy.num_actual_experience-1):
                    self.metric[f'Forgetting/Eval/Exp{strategy.num_actual_experience}'] += self.metric[f'Forgetting/Eval/Exp{strategy.num_actual_experience}/Exp{i+1}']
                self.metric[f'Forgetting/Eval/Exp{strategy.num_actual_experience}'] /= (strategy.num_actual_experience - 1)

            elif status == 'after_training':
                self.metric[f'AvgForgetting'] = 0
                for i in range(1, strategy.num_actual_experience):
                    for j in range(1, i+1):
                        self.metric[f'AvgForgetting'] += self.metric[f'Forgetting/Eval/Exp{i+1}/Exp{j}']
                self.metric[f'AvgForgetting'] /= (strategy.num_actual_experience*(strategy.num_actual_experience-1) / 2)

            else:
                pass