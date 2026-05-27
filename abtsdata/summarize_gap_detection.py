import numpy as np

from psidata.manager import add_default_options, process_files

from . import _summarize_behavior as behavior


glob_gonogo = '**/*gap-detection*'


def gap_fmt_settings(frequency, n_gap):
    return f'{n_gap} gaps, {frequency}Hz'


def gap_process_trial_log(tl):
    # Early versions did not have n_gap.
    if 'n_gap' not in tl:
        tl['n_gap'] = 1
    return tl


def gap_process_folder(path, manager, glob_pattern, yes_resp):
    behavior.process_folder(
        path,
        manager,
        glob_pattern,
        grouping=['frequency', 'n_gap'],
        fmt_settings_cb=gap_fmt_settings,
        process_trial_log_cb=gap_process_trial_log,
        test_param='gap',
        test_param_label='Gap duration (s)',
        yes_resp=yes_resp,
        psychometric_fit_kw={
            'core': 'poly',
            'sigmoid': 'exponential',
            'alpha': {'dist': 'HalfNormal', 'sigma': 1},
            'beta': {'dist': 'LogNormal', 'mu': 1},
        },
    )


def main_date_gonogo():
    import argparse
    parser = argparse.ArgumentParser('summarize-date-gap-gonogo')
    add_default_options(parser)
    args = vars(parser.parse_args())
    process_files(glob_gonogo,
                  lambda *args, **kw: gap_process_folder(*args, **kw, yes_resp='resp_1', glob_pattern=glob_gonogo),
                  expected_suffixes=behavior.expected_suffixes,
                  pathfinder=behavior.date_pathfinder,
                  manager_kw=dict(file_template=lambda path: f'{path.stem} {path.parent.stem} daily gap-detection'),
                  **args)


def main_experiment_gonogo():
    import argparse
    parser = argparse.ArgumentParser('summarize-gap-gonogo')
    add_default_options(parser)
    args = vars(parser.parse_args())
    process_files(glob_gonogo,
                  lambda *args, **kw: gap_process_folder(*args, **kw, yes_resp='resp_1', glob_pattern=glob_gonogo),
                  expected_suffixes=behavior.expected_suffixes,
                  **args)
