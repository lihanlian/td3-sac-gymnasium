from collections import defaultdict #???
import json
import os
import csv
import shutil
import torch
import numpy as np
# from termcolor import colored

COMMON_TRAIN_FORMAT = [
    ('episode', 'E', 'int'),
    ('step', 'S', 'int'),
    ('episode_reward', 'R', 'float'),
]

COMMON_EVAL_FORMAT = [
    ('episode', 'E', 'int'),
    ('step', 'S', 'int'),
    ('episode_reward', 'R', 'float')
]

AGENT_TRAIN_FORMAT = {
    'sac': [
        ('batch_reward', 'BR', 'float'),
        ('actor_loss', 'ALoss', 'float'),
        ('critic_loss', 'CLoss', 'float'),
        ('alpha_loss', 'TLoss', 'float'),
        ('alpha_value', 'TVAL', 'float'),
        ('actor_entropy', 'AENT', 'float')
    ],

    'td3': [
        ('batch_reward', 'BR', 'float'),
        ('actor_loss', 'ALoss', 'float'),
        ('critic_loss', 'CLoss', 'float'),
        ('alpha_loss', 'TLoss', 'float'),
        ('alpha_value', 'TVAL', 'float'),
        ('actor_entropy', 'AENT', 'float')
    ]
}

class AverageMeter():
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)
    
class MetersGroup():
    def __init__(self, file_name, formating):
        self._csv_file_name = self._prepare_file(file_name, 'csv')
        self._formating = formating
        self._meters = defaultdict(AverageMeter)
        self._csv_file = open(self._csv_file_name, 'w')
        self._csv_writer = None

    def _prepare_file(self, prefix, suffix):
        file_name = f'{prefix}.{suffix}'
        if os.path.exists(file_name):
            os.remove(file_name)

        return file_name
    
    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith('train'):
                key = key[len('train') + 1:]
            else:
                key = key[len('eval') + 1:]
            key = key.replace('/', '_')
            data[key] = meter.value()
        return data
    
    def _dump_to_csv(self, data):
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(self._csv_file,
                                              fieldnames=sorted(data.keys()),
                                              restval=0.0)
            self._csv_writer.writeheader()

        self._csv_writer.writerow(data)
        self._csv_file.flush()

    def _format(self, key, value, ty):
        if ty == 'int':
            value = int(value)
            return f'{key}: {value}'
        elif ty == 'float':
            return f'{key}: {value:.04f}'
        elif ty == 'time':
            return f'{key}: {value:04.1f} s'
        else:
            raise f'invalid format type: {ty}'
        
    def _dumpt_to_console(self, data, prefix):
        # prefix = colored(prefix, 'yellow' if prefix == 'train' else 'green')
        # pieces = [f'| {prefix: < 14}']
        pieces = []
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print(' | '.join(pieces))

    def dump(self, step, prefix, save=True):
        if len(self._meters) == 0:
            return
        if save:
            data = self._prime_meters()
            data['step'] = step
            self._dump_to_csv(data)
            self._dumpt_to_console(data,prefix)
        # !what is the use of clear()???
        self._meters.clear()

class Logger():
    def __init__(self, log_dir, log_freq=10000, agent='sac'):
        self._log_dir = log_dir
        self._log_freq = log_freq
        assert agent in AGENT_TRAIN_FORMAT
        train_format = COMMON_TRAIN_FORMAT + AGENT_TRAIN_FORMAT[agent]
        self._train_mg = MetersGroup(os.path.join(log_dir, 'train'),
                                     formating=train_format)
        self._eval_mg = MetersGroup(os.path.join(log_dir, 'eval'),
                                    formating=COMMON_EVAL_FORMAT)
        
    def _should_log(self, step, log_freq):
        log_freq = log_freq or self._log_freq
        return step % log_freq == 0
    
    def log(self, key, value, step, n=1, log_freq=1):
        if not self._should_log(step, log_freq):
            return
        assert key.startswith('train') or key.startswith('eval')
        if type(value) == torch.Tensor:
            value = value.item()
        mg = self._train_mg if key.startswith('train') else self._eval_mg
        mg.log(key, value, n)

    def dump(self, step, save=True, ty=None):
        if ty is None:
            self._train_mg.dump(step, 'train', save)
            self._eval_mg.dump(step, 'eval', save)
        elif ty == 'eval':
            self._eval_mg.dump(step, 'eval', save)
        elif ty == 'train':
            self._train_mg.dump(step, 'train', save)
        else:
            raise f'invalide log type: {ty}'