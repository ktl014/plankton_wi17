import os
from visualdl import LogWriter
from utils.constants import *


class Logger(object):
    logger = None

    def __init__(self, name, log_dir, resume, phases=(TRAIN, VALID), step=10):
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        if Logger.logger is None:
            Logger.logger = LogWriter(log_dir, sync_cycle=10)

        self.scalars = {}
        self.count = {phase: 0 for phase in phases}
        self.fhandle = {phase: open(os.path.join(log_dir, '{}_{}.txt'.format(phase, name)), 'a') for phase in phases}
        self.step = step

        for phase in phases:
            with self.logger.mode(phase):
                self.scalars[phase] = self.logger.scalar('scalars/{}_{}'.format(phase, name))

            if resume:
                f = open(os.path.join(log_dir, '{}_{}.txt'.format(phase, name)), 'r')
                for i, record in enumerate(f):
                    if (i % 10 == 0 or phase != TRAIN) and len(record) > 1:
                        self.scalars[phase].add_record(i, float(record))
                    self.count[phase] += 1
                print('=>     {} {} {} data points loaded'.format(phase, name, self.count[phase]))
                f.close()

    def add_record(self, phase, record):
        if phase != TRAIN or self.count[phase] % 10 == 0:
            self.scalars[phase].add_record(self.count[phase], record)
        self.fhandle[phase].write('{:.6f}\n'.format(record))
        self.count[phase] += 1

    @staticmethod
    def write_meta(path, args):
        meta_file = open(os.path.join(path, 'meta.txt'), 'a')
        for arg in vars(args):
            meta_file.write('{}: {}\n'.format(arg, getattr(args, arg)))
        meta_file.write('\n\n')

    @staticmethod
    def read_meta(meta_file):
        class DotDict(dict):
            __getattr__ = dict.get
            __setattr__ = dict.__setitem__
            __delattr__ = dict.__delitem__

        args = DotDict()
        for line in open(meta_file, 'r').read().splitlines():
            if line:
                key, value = line.split(': ')
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                args[key] = value
        return args

