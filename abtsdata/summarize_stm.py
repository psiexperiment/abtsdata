import numpy as np

from psidata.manager import add_default_options, process_files

from . import _summarize_behavior as behavior


glob_2AFC = '**/*modulation-2AFC*'
glob_gonogo = '**/*modulation-gonogo*'


def stm_fmt_settings(cpo, cps, fc, ml, mg, cl):
    t = f'{cps}Hz, {cpo}c/o, {fc}kHz, {cl}dB SPL'
    if np.isfinite(ml):
        t = f'{t} {ml} dB SPL mask'
    if np.isfinite(mg):
        t = f'{t} {mg} dB flank'
    return t


def stm_process_trial_log(tl):
    # Early experiments did not have trial_subtype.
    if 'trial_subtype' not in tl:
        tl['trial_subtype'] = 'unknown'
    if 'masker_gain' not in tl:
        tl['masker_gain'] = -np.inf

    # Early versions of program coded increasing depth as positive, when it
    # should be negative (0 indicates unmodulated, with negative values
    # indicating increasing depth).
    if tl['stm_depth'].min() == 0:
        tl['stm_depth'] = -tl['stm_depth']
    if 'masker_level' not in tl:
        tl['masker_level'] = -np.inf

    return tl


def stm_process_folder(path, manager, glob_pattern, yes_resp='resp_2'):
    behavior.process_folder(
        path,
        manager,
        glob_pattern,
        grouping=['cpo', 'cps', 'fc', 'masker_level', 'masker_gain', 'center_level'],
        fmt_settings_cb=stm_fmt_settings,
        process_trial_log_cb=stm_process_trial_log,
        test_param='stm_depth',
        test_param_label='STM depth (dB)',
        yes_resp=yes_resp,
        psychometric_fit_kw={
            'core': 'ab',
            'sigmoid': 'logistic',
            'alpha': {'dist': 'Normal', 'mu': 0, 'sigma': 10},
            'beta': {'dist': 'LogNormal', 'mu': 1},
            'slope': 'negative',
        },
    )


def main_date_2AFC():
    import argparse
    parser = argparse.ArgumentParser('summarize-date-modulation-2AFC')
    add_default_options(parser)
    args = vars(parser.parse_args())
    process_files(glob_2AFC,
                  lambda *args, **kw: stm_process_folder(*args, **kw, yes_resp='resp_2', glob_pattern=glob_2AFC),
                  expected_suffixes=behavior.expected_suffixes,
                  pathfinder=behavior.date_pathfinder,
                  manager_kw=dict(file_template=lambda path: f'{path.stem} {path.parent.stem} daily modulation-2AFC'),
                  **args)


def main_experiment_2AFC():
    import argparse
    parser = argparse.ArgumentParser('summarize-modulation-2AFC')
    add_default_options(parser)
    args = vars(parser.parse_args())
    process_files(glob_2AFC,
                  lambda *args, **kw: stm_process_folder(*args, **kw, yes_resp='resp_2', glob_pattern=glob_2AFC),
                  expected_suffixes=behavior.expected_suffixes,
                  **args)


def main_date_gonogo():
    import argparse
    parser = argparse.ArgumentParser('summarize-date-modulation-gonogo')
    add_default_options(parser)
    args = vars(parser.parse_args())
    process_files(glob_gonogo,
                  lambda *args, **kw: stm_process_folder(*args, **kw, yes_resp='resp_1', glob_pattern=glob_gonogo),
                  expected_suffixes=behavior.expected_suffixes,
                  pathfinder=behavior.date_pathfinder,
                  manager_kw=dict(file_template=lambda path: f'{path.stem} {path.parent.stem} daily modulation-gonogo'),
                  **args)


def main_experiment_gonogo():
    import argparse
    parser = argparse.ArgumentParser('summarize-modulation-gonogo')
    add_default_options(parser)
    args = vars(parser.parse_args())
    process_files(glob_gonogo,
                  lambda *args, **kw: stm_process_folder(*args, **kw, yes_resp='resp_1', glob_pattern=glob_gonogo),
                  expected_suffixes=behavior.expected_suffixes,
                  **args)
