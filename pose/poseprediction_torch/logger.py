import os
from visualdl import LogWriter, LogReader
from utils.constants import *


class Logger(object):
    logger = None

    def __init__(self, name, log_dir, resume, phases=(TRAIN, VALID)):
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        if Logger.logger is None:
            Logger.logger = LogWriter(log_dir, sync_cycle=10)
        self.scalars = {}
        self.count = {phase: 0 for phase in phases}

        for phase in phases:
            if resume:
                reader = LogReader(log_dir)
                records = reader.scalar('scalars/{}_{}'.format(phase, name)).records()
                self.n = len(records)

            with self.logger.mode(phase):
                self.scalars[phase] = self.logger.scalar('scalars/{}_{}'.format(phase, name))

            if resume:
                for i, record in enumerate(records):
                    self.scalars[phase].add_record(i, record)

    def add_record(self, phase, record):
        self.scalars[phase].add_record(self.count[phase], record)
        self.count[phase] += 1

    @staticmethod
    def write_meta(path, args):
        meta_file = open(os.path.join(path, 'meta.txt'), 'a')
        for arg in vars(args):
            meta_file.write('{}: {}\n'.format(arg, getattr(args, arg)))
        meta_file.write('\n\n')
