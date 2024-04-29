from logging.handlers import RotatingFileHandler
import time
import os
import logging
import multiprocess
import warnings
import difflib
import yaml
import numpy as np
from tqdm import tqdm
from anytree import NodeMixin, RenderTree


class Timer:
    def __init__(self, files=[1], prefix="Imported in"):
        if isinstance(files, int):
            files = [files]
        self.len = len(files)
        self.prefix = prefix

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        if self.len == 1:
            print(f'{self.prefix} {elapsed:.6f} seconds')
        else:
            print(f'{self.prefix} {elapsed:.6f} seconds ({self.len/elapsed:.0f}/s)')


class Colors:
    RED = '\033[91m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    BOLD = '\033[1m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def get_color(index):
    colors = [x for x in Colors.__dict__ if '_' not in x]
    colors = colors[:7]
    name = colors[index]
    return getattr(Colors, name)


def generic_repr(obj) -> str:
    name = type(obj).__name__
    vars_list = [f'{key}={val!r}' for key, val in vars(obj).items()]
    vars_str = ', '.join(vars_list)
    return f'{name}({vars_str})'


def autoset_n_jobs(fn):
    def wrapped(*args, **kwargs):
        if 'njobs' not in kwargs or kwargs['njobs'] is None:
            kwargs['njobs'] = multiprocess.cpu_count()
        return fn(*args, **kwargs)
    return wrapped


class CustomNode(NodeMixin):
    def __init__(self, parent, node_type, identifier, modality=False, file_count=1, level=0):
        super().__init__()
        self.parent = parent
        self.node_type = node_type
        if identifier == 'file::memory:?cache=shared':
            self.identifier = 'memory'
        else:
            self.identifier = identifier
        self.modality = modality
        self.file_count = file_count
        self.level = level

    def __repr__(self):
        out = f'{self._wrap(self.node_type)}: '
        if self.modality:
            out += f'{Colors.ITALIC}{Colors.UNDERLINE}{self.identifier}{Colors.END}'
            out += f' with {Colors.ITALIC}{Colors.RED}{self.file_count}{Colors.END} file'
            if self.file_count > 1:
                out += 's'
        else:
            out += self.identifier
        return out

    def __str__(self):
        output = []
        for pre, fill, node in RenderTree(self, childiter=self._str_sort):
            output.append(pre+repr(node))
        return '\n'.join(output)

    def _wrap(self, text):
        color = get_color(self.level)
        return f'{color}{text}{Colors.END}'

    def _str_sort(self, items: list) -> list:
        return sorted(items, key=lambda item: item.identifier)


def series_df_generator(frame_df, modalities=None, volume_series_uids=None):
    for series_uid in frame_df.SeriesInstanceUID.unique():
        if volume_series_uids is not None and series_uid in volume_series_uids:
            continue
        series_df = frame_df[frame_df.SeriesInstanceUID == series_uid]
        modality = series_df.iloc[0]['Modality']
        if modalities is None or modality in modalities:
            yield series_df


def make_log(logfile, db_name):
    # Make log file
    log = logging.getLogger(logfile)
    log.setLevel(logging.INFO)

    # Make and add handler
    handler = RotatingFileHandler(logfile, mode='w', maxBytes=5*1024*1024,
                                  backupCount=2, encoding=None, delay=0)
    handler.setLevel(logging.INFO)
    log_fmt = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                                datefmt='%Y-%m-%d:%H:%M:%S')
    handler.setFormatter(log_fmt)
    log.addHandler(handler)
    log.info(f'Cohort initialized at {db_name}')
    return log


def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    relative = 'DICOManager/' + os.path.basename(filename)
    return f'{relative}: line {lineno}: {category.__name__}: {message} \n'


def log_notification(prefix='DICOManager'):
    def outer(fn):
        def inner(cls, *args, **kwargs):
            out = fn(cls, *args, **kwargs)
            start = cls._log_n_lines
            current = line_count(cls.logfile)
            if current - start:
                logpath = cls._logger.handlers[0].baseFilename
                warnings.formatwarning = warning_on_one_line
                warnings.warn(f'{prefix} added notifications to the logfile at {logpath}')
            cls._log_n_lines = current
            return out
        return inner
    return outer


def line_count(fname):
    def _gen(reader):
        b = reader(2 ** 16)
        while b:
            yield b
            b = reader(2 ** 16)

    with open(fname, "rb") as f:
        count = sum(buf.count(b"\n") for buf in _gen(f.raw.read))
    return count


def unnest(nested):
    out = []
    for n in nested:
        if not len(n):
            continue
        elif isinstance(n[0], list):
            out.extend(unnest(n))
        else:
            out.extend(n)
    return out


def save_dir_check(fn):
    def wrapped(cls, *args, **kwargs):
        if kwargs.get('save_dir', None) is None:
            if not hasattr(cls, 'save_dir') or cls.save_dir is None:
                raise IOError('No save directory specified or already used for Cohort')
            else:
                kwargs['save_dir'] = cls.save_dir
        return fn(cls, *args, **kwargs)
    return wrapped


def split_sops(sops):
    index = len(sops[0])
    for sop in sops[1:]:
        sm = difflib.SequenceMatcher(None, sops[0], sop)
        limit = sm.find_longest_match().size
        if limit < index:
            index = limit

    root = sops[0][:index]
    tails = [x[index:] for x in sops]

    if not len(tails):
        tails = None

    return (root, tails)


def read_yaml(filename):
    with open(filename, 'r') as yf:
        contents = yaml.safe_load(yf)
    if '_comment' in contents:
        del contents['_comment']
    return contents


@autoset_n_jobs
def apply_fn_parallelized(fn, c, n_in_group=2**8, njobs=None, progress_bar=None, total=None):
    def inner(pbar=None):
        output = []
        with multiprocess.Pool(processes=njobs) as P:
            while True:
                groups = []
                for _ in range(njobs):
                    x = c.fetchmany(n_in_group)
                    if pbar:
                        pbar.update(len(x))
                    if len(x) == 0:
                        return output
                    groups.append(x)
                output.extend(list(P.imap_unordered(fn, groups)))

    if progress_bar and total:
        with tqdm(total=total) as pbar:
            pbar.set_description('Saving')
            return inner(pbar)
    return inner()


def to_16bit(array):
    norm = (array - np.min(array)) / (np.max(array) - np.min(array))
    arr16 = np.int16(np.round(norm * 1024))
    return arr16
