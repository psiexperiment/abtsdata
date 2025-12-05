from fnmatch import fnmatch
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from psidata.api import Recording
from psidata.manager import add_default_options, process_files

from abtsdata import psychometric


GLOB_PATTERN = '**/*modulation-2AFC*'


expected_suffixes = [
    'performance.pdf',
    'stats.json',
    'performance.csv',
    'performance_fit.csv',
    'threshold.csv'
]


def fmt_settings(cpo, cps, fc, ml, cl):
    return f'{cps} Hz, {cpo} c/o, {fc} kHz, {cl} dB SPL, {ml} dB SPL masker'


def process_folder(path, manager):
    path = Path(path)

    figure, axes = plt.subplot_mosaic(
        [['resp', 'resp', 'resp', 'tc'],
        ['tpm', 'tpm', 'tpm', '.'],
        ['tpm', 'tpm', 'tpm', '.'],
        ['rt', 'n', 'perf', 'd_prime'],
        ['rt', 'n', 'perf', 'd_prime']],
        figsize=(11, 7), layout='constrained')
    axes['resp'].sharex(axes['tpm'])
    axes['resp'].sharey(axes['tc'])

    trial_logs = []
    n_pellets = 0

    if path.suffix == '.zip':
        filenames = [path]
    else:
        filenames = [path for path in path.glob(GLOB_PATTERN) \
                     if path.suffix == '.zip' and '_exclude' not in str(path)]

    for filename in filenames:
        ds = Recording(filename)
        try:
            tl = ds.trial_log.copy()
            n_pellets += ds.event_log['event'].str.startswith('deliver_').sum()
            datetime, _ = filename.stem.split(' ', 1)
            tl['datetime'] = datetime
            tl['trial'] = range(len(tl))
            trial_logs.append(tl)
        except (AttributeError, pd.errors.EmptyDataError):
            # No trials recorded. Either trial log is empty or event log/trial
            # log are missing.
            pass

    if len(trial_logs) == 0:
        # No data found. Create sentinel files so this experiment does not get
        # reprocessed.
        manager.save_fig(figure, 'performance.pdf')
        manager.save_df(pd.DataFrame(), 'performance.csv')
        manager.save_df(pd.DataFrame(), 'performance_fit.csv')
        stats = {
            'n_trials': 0,
            'tpm_average': 0,
            'n_pellets': n_pellets,
        }
        manager.save_dict(stats, 'stats.json')

        # Create an empty placeholder for the trace file.
        Path(manager.get_proc_filename('trace.nc')).touch()
        print('returning short')
        return

    tl = pd.concat(trial_logs).set_index(['datetime', 'trial'], verify_integrity=True).sort_index()
    tl = tl.reset_index()

    tl['yes'] = tl['response'] == 'resp_2'
    grouping = ['cpo', 'cps', 'fc', 'masker_level', 'center_level']

    # Early versions of program coded increasing depth as positive, when it
    # should be negative (0 indicates unmodulated, with negative values
    # indicating increasing depth).
    if tl['stm_depth'].min() == 0:
        tl['stm_depth'] = -tl['stm_depth']
    if 'masker_level' not in tl:
        tl['masker_level'] = -np.inf

    # Response details (e.g., 1, 2, no response)
    resp = {
        'no_response': {'color': '0.5', 'value': 0},
        'resp_1': {'color': 'k', 'value': -1},
        'resp_2': {'color': 'k', 'value': 1},
    }
    r = tl.groupby('response').size().rename('size')
    for r_name, r_info in resp.items():
        x = tl.query('response == @r_name')
        axes['resp'].plot(x.index, np.ones_like(x.index) * r_info['value'], 'o', color=r_info['color'], mec='w', mew=1)
        if r_name in r:
            axes['tc'].barh(r_info['value'], r[r_name], color=r_info['color'])
    axes['resp'].yaxis.set_ticks([-1, 0, 1])
    axes['resp'].yaxis.set_ticklabels(['Resp. 1', 'No resp.', 'Resp. 2'])
    axes['tc'].set_xlabel('Number of Trials')

    # Trials per minute
    tpm = 60 / tl['trial_start'].diff()
    axes['tpm'].plot(tpm, label='Per trial')
    axes['tpm'].plot(tpm.rolling(5).mean(), label='Moving average')
    axes['tpm'].set_xlabel('Trial Number')
    axes['tpm'].set_ylabel('Trials per minute')
    axes['tpm'].legend(bbox_to_anchor=(1, 1), loc='upper right')

    m = tl['datetime'] != tl['datetime'].shift(1)
    indices = m.loc[m].index.values
    for ax_name in ('tpm', 'resp'):
        ax = axes[ax_name]
        for i in indices:
            ax.axvline(i, ls=':', color='k')
            if ax_name == 'resp':
                continue
            cpo = tl.loc[i, 'cpo']
            cps = tl.loc[i, 'cps']
            fc = tl.loc[i, 'fc']
            ml = tl.loc[i, 'masker_level']
            cl = tl.loc[i, 'center_level']
            key = tl.loc[i, grouping]
            t = fmt_settings(*key)
            text = ax.text(i, 1.05, t, transform=ax.get_xaxis_transform(), fontsize='x-small')
            # If it's part of the constrained layout calculations, it sometimes
            # squishes the next column of axes if the text spills off the right
            # edge of the axes.
            text.set_in_layout(False)

    # Plot the reaction time
    rt = tl.groupby(grouping + ['stm_depth'])['response_time'].agg(['mean', 'sem'])
    colors = {}
    legend = {}
    for key, rt_subset in rt.groupby(grouping, group_keys=False):
        label = fmt_settings(*key)
        x = rt_subset.index.get_level_values('stm_depth')
        p = axes['rt'].errorbar(x, rt_subset['mean'], yerr=rt_subset['sem'], fmt='o-')
        legend[label] = p
        colors[key] = p[0].get_color()
        axes['rt'].set_xlabel('STM depth (dB)')
        axes['rt'].set_ylabel('Reaction time (sec)')

    stats = {
        'n_trials': len(tpm),
        'tpm_average': tpm.mean(),
        'n_pellets': n_pellets,
    }

    if 'trial_subtype' not in tl:
        # Don't attempt to estimate performance because we can't differentiate
        # between repeat and non-repeat trials. Save dummy dataframes for the
        # behavior performance. This will create files that ensure that this
        # experiment is skipped over when reprocessing. Since these are early
        # experiments, there is no need to try and recover some data from
        # these.
        manager.save_fig(figure, 'performance.pdf')
        manager.save_dict(stats, 'stats.json')
        manager.save_df(pd.DataFrame(), 'performance.csv', index=True)
        manager.save_df(pd.DataFrame(), 'performance_fit.csv', index=True)
        manager.save_df(pd.DataFrame(), 'threshold.csv', index=True)
        return

    agg = {'p': 'mean', 'k': 'sum', 'n': 'size'}
    summary_all = tl.groupby(grouping + ['stm_depth'])['yes'].agg(**agg).add_prefix('all_')
    summary = tl.query('trial_subtype != "repeat"').groupby(grouping + ['stm_depth'])['yes'].agg(**agg)
    summary = summary.join(summary_all)

    results = {}
    for i, (key, summary_subset) in enumerate(summary.groupby(grouping, group_keys=False)):
        summary_subset = summary_subset.reset_index(grouping, drop=True)

        # Plot the number of trials per depth. Don't show 0 since it has so many
        # trials compared to the other depths.
        n = summary_subset.query('stm_depth != 0')['all_n']
        axes['n'].plot(n, 'o-', label='All trials', color=colors[key])
        n = summary_subset.query('stm_depth != 0')['n']
        axes['n'].plot(n, 'o:', label='Exl. repeats', color=colors[key])
        axes['n'].set_xlabel('STM depth (dB)')
        axes['n'].set_ylabel('# of trials')

        if 0 not in summary_subset.index:
            # If we switched to a new setting, the animal may not have
            # performed any reference (0) trials.
            results.setdefault('summary', {})[key] = summary_subset
        elif len(summary_subset) < 4:
            # Not enough depths to fit a meaningful psychometric function.
            results.setdefault('summary', {})[key] = summary_subset
        else:
            result = psychometric.fit_psychometric(summary_subset)
            text_y = (i + 1) * 0.1
            psychometric.plot_psychometric(**result, which='p', ax=axes['perf'], color=colors[key], text_y=text_y, show_ci=False)
            psychometric.plot_psychometric(**result, which='d', ax=axes['d_prime'], color=colors[key], text_y=text_y, show_ci=False)
            for k, v in result.items():
                results.setdefault(k, {})[key] = v

    results = {k: pd.concat(v, names=grouping) for k, v in results.items() if k != 'trace'}

    legend = axes['d_prime'].legend(legend.values(), legend.keys(), bbox_to_anchor=(1, 1), loc='lower right', fontsize='x-small')
    legend.set_in_layout(False)
    manager.save_fig(figure, 'performance.pdf')
    manager.save_dict(stats, 'stats.json')
    manager.save_df(results['summary'], 'performance.csv', index=True)
    if 'fit' in results:
        manager.save_df(results['fit'], 'performance_fit.csv', index=True)
        manager.save_df(results['threshold'].unstack(), 'threshold.csv', index=True)


def date_pathfinder(folder, glob_pattern):
    seen = []
    for (dirname, dirnames, filenames) in os.walk(folder):
        for filename in filenames:
            if '_exclude' in str(filename):
                continue
            if not filename.endswith('.zip'):
                continue
            full_filename = os.path.join(dirname, filename)
            if fnmatch(full_filename, glob_pattern):
                key = Path(os.path.split(dirname)[0])
                if key in seen:
                    continue
                else:
                    seen.append(key)
                    yield key


def main_date():
    import argparse
    parser = argparse.ArgumentParser('summarize-date-modulation-2AFC')
    add_default_options(parser)
    args = vars(parser.parse_args())
    process_files(GLOB_PATTERN, process_folder,
                  expected_suffixes=expected_suffixes,
                  pathfinder=date_pathfinder, **args)


def main_experiment():
    import argparse
    parser = argparse.ArgumentParser('summarize-modulation-2AFC')
    add_default_options(parser)
    args = vars(parser.parse_args())
    process_files('**/*modulation-2AFC*', process_folder,
                  expected_suffixes=expected_suffixes, **args)
