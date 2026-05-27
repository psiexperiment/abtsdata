from fnmatch import fnmatch
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import palettable.scientific.sequential
import pandas as pd
from scipy import stats

from psidata.api import Recording
from psidata.manager import add_default_options, process_files

from abtsdata import dataset, psychometric
from abtsdata.plot import add_weekends

################################################################################
# Code for analyzing single experiment
################################################################################
expected_suffixes = [
    'performance.pdf',
    'stats.json',
    'performance.csv',
    'performance_fit.csv',
    'threshold.csv'
]


def load_trial_logs(path, glob_pattern, process_trial_log_cb=None):
    if process_trial_log_cb is None:
        process_trial_log_cb = lambda x: x

    path = Path(path)
    if path.suffix == '.zip':
        filenames = [path]
    else:
        filenames = [path for path in path.glob(glob_pattern) \
                     if path.suffix == '.zip' and '_exclude' not in str(path)]

    trial_logs = []
    n_pellets = 0

    for filename in filenames:
        ds = Recording(filename)
        try:
            tl = process_trial_log_cb(ds.trial_log.copy())
            n_pellets += ds.event_log['event'].str.startswith('deliver_').sum()
            datetime, _ = filename.stem.split(' ', 1)
            tl['date'], tl['time'] = datetime.split('-')
            tl['datetime'] = datetime
            tl['trial'] = range(len(tl))
            trial_logs.append(tl)
        except (AttributeError, pd.errors.EmptyDataError):
            # No trials recorded. Either trial log is empty or event log/trial
            # log are missing.
            pass

    cols = ['date', 'time', 'datetime', 'trial', 'trial_subtype']
    if len(trial_logs) == 0:
        tl = pd.DataFrame(columns=cols)
    else:
        tl = pd.concat(trial_logs)
    tl = tl.set_index(cols, verify_integrity=True).sort_index()
    return tl, n_pellets


def process_folder(path, manager, glob_pattern, grouping, fmt_settings_cb,
                   test_param, test_param_label, yes_resp='resp_2',
                   process_trial_log_cb=None, psychometric_fit_kw=None, x_scale='linear'):
    path = Path(path)

    if psychometric_fit_kw is None:
        psychometric_fit_kw = {}

    figure, axes = plt.subplot_mosaic(
        [['resp', 'resp', 'resp', 'tc'],
        ['tpm', 'tpm', 'tpm', '.'],
        ['tpm', 'tpm', 'tpm', '.'],
        ['fa', 'fa', 'fa', '.'],
        ['fa', 'fa', 'fa', '.'],
        ['rt', 'n', 'perf', 'd_prime'],
        ['rt', 'n', 'perf', 'd_prime']],
        figsize=(11, 9), layout='constrained')
    axes['resp'].sharex(axes['tpm'])
    axes['resp'].sharey(axes['tc'])

    trial_logs, n_pellets = load_trial_logs(path, glob_pattern,
                                            process_trial_log_cb)
    tl = trial_logs.query('trial_subtype != "remind"').reset_index()

    if len(tl) == 0:
        # No data found. Create sentinel files so this experiment does not get
        # reprocessed.
        manager.save_fig(figure, 'performance.pdf')
        manager.save_df(pd.DataFrame(), 'performance.csv')
        manager.save_df(pd.DataFrame(), 'performance_fit.csv')
        exp_stats = {
            'n_trials': 0,
            'tpm_average': 0,
            'n_pellets': n_pellets,
        }
        manager.save_dict(exp_stats, 'stats.json')

        # Create an empty placeholder for the trace file.
        Path(manager.get_proc_filename('trace.nc')).touch()
        return

    tl['yes'] = tl['response'] == yes_resp

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

    for window in (10, 20, 40):
        fa = tl.query(f'{test_param} == 0')['yes'].rolling(window=window).mean()
        axes['fa'].plot(fa, label=f'Last {window}')
    axes['fa'].set_xlabel('Trial Number')
    axes['fa'].set_ylabel('False alarm rate')
    axes['fa'].legend(bbox_to_anchor=(0, 1), loc='upper left')

    m = tl['datetime'] != tl['datetime'].shift(1)
    indices = m.loc[m].index.values
    for ax_name in ('tpm', 'resp'):
        ax = axes[ax_name]
        for i in indices:
            ax.axvline(i, ls=':', color='k')
            if ax_name == 'resp':
                continue

            row = tl.iloc[i]
            key = [row[c] for c in grouping]
            t = fmt_settings_cb(*key)
            text = ax.text(i, 1.05, t, transform=ax.get_xaxis_transform(), fontsize='x-small')
            # If it's part of the constrained layout calculations, it sometimes
            # squishes the next column of axes if the text spills off the right
            # edge of the axes.
            text.set_in_layout(False)

    # Plot the reaction time
    rt = tl.groupby(grouping + [test_param])['response_time'].agg(['mean', 'sem'])
    colors = {}
    legend = {}
    for key, rt_subset in rt.groupby(grouping, group_keys=False):
        label = fmt_settings_cb(*key)
        x = rt_subset.index.get_level_values(test_param)
        p = axes['rt'].errorbar(x, rt_subset['mean'], yerr=rt_subset['sem'], fmt='o-')
        legend[label] = p
        colors[key] = p[0].get_color()
        axes['rt'].set_xlabel(test_param_label)
        axes['rt'].set_ylabel('Reaction time (sec)')

    exp_stats = {
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
        manager.save_dict(exp_stats, 'stats.json')
        manager.save_df(pd.DataFrame(), 'performance.csv', index=True)
        manager.save_df(pd.DataFrame(), 'performance_fit.csv', index=True)
        manager.save_df(pd.DataFrame(), 'threshold.csv', index=True)
        return

    agg = {'p': 'mean', 'k': 'sum', 'n': 'size'}
    summary_all = tl.groupby(grouping + [test_param])['yes'].agg(**agg).add_prefix('all_')
    summary = tl.query('trial_subtype != "repeat"').groupby(grouping + [test_param])['yes'].agg(**agg)
    summary = summary.join(summary_all)

    results = {}
    for i, (key, summary_subset) in enumerate(summary.groupby(grouping, group_keys=False)):
        summary_subset = summary_subset.reset_index(grouping, drop=True)

        # Plot the number of trials per depth. Don't show 0 since it has so many
        # trials compared to the other depths.
        n = summary_subset.query(f'{test_param} != 0')['all_n']
        axes['n'].plot(n, 'o-', label='All trials', color=colors[key])
        n = summary_subset.query(f'{test_param} != 0')['n']
        axes['n'].plot(n, 'o:', label='Exl. repeats', color=colors[key])
        axes['n'].set_xlabel(test_param_label)
        axes['n'].set_ylabel('# of trials')

        if 0 in summary_subset.index:
            p_clipped = summary_subset['p'].clip(0.05, 0.95)
            summary_subset['d'] = stats.norm.ppf(p_clipped) - stats.norm.ppf(p_clipped.loc[0])
            p_clipped = summary_subset['all_p'].clip(0.05, 0.95)
            summary_subset['all_d'] = stats.norm.ppf(p_clipped) - stats.norm.ppf(p_clipped.loc[0])
        else:
            summary_subset['d'] = np.nan
            summary_subset['all_d'] = np.nan

        text_y = (i + 1) * 0.1

        if 0 not in summary_subset.index:
            # If we switched to a new setting, the animal may not have
            # performed any reference (0) trials. Cannot calculate performance.
            results.setdefault('summary', {})[key] = summary_subset
        elif len(summary_subset) < 4:
            # Not enough depths to fit a meaningful psychometric function.
            results.setdefault('summary', {})[key] = summary_subset
            psychometric.plot_psychometric(summary=summary_subset, which='p',
                                           ax=axes['perf'], color=colors[key],
                                           text_y=text_y, show_ci=False,
                                           x_label=test_param_label)

            psychometric.plot_psychometric(summary=summary_subset, which='d',
                                           ax=axes['d_prime'],
                                           color=colors[key], text_y=text_y,
                                           show_ci=False,
                                           x_label=test_param_label)
        else:
            result = psychometric.fit_psychometric(summary_subset,
                                                   **psychometric_fit_kw)

            psychometric.plot_psychometric(**result, which='p',
                                           ax=axes['perf'], color=colors[key],
                                           text_y=text_y, show_ci=False,
                                           x_label=test_param_label)
            psychometric.plot_psychometric(**result, which='d',
                                           ax=axes['d_prime'],
                                           color=colors[key], text_y=text_y,
                                           show_ci=False,
                                           x_label=test_param_label)
            for k, v in result.items():
                results.setdefault(k, {})[key] = v

    results = {k: pd.concat(v, names=grouping) for k, v in results.items() if k != 'trace'}

    legend = axes['d_prime'].legend(legend.values(), legend.keys(), bbox_to_anchor=(1, 1), loc='lower right', fontsize='x-small')
    legend.set_in_layout(False)
    manager.save_fig(figure, 'performance.pdf')
    manager.save_dict(exp_stats, 'stats.json')
    manager.save_df(results['summary'], 'performance.csv', index=True)
    if 'fit' in results:
        manager.save_df(results['fit'], 'performance_fit.csv', index=True)
        manager.save_df(results['threshold'].unstack(), 'threshold.csv', index=True)
    else:
        # Create placeholder files to prevent this experiment from being
        # reprocessed.
        manager.save_df(pd.DataFrame(), 'performance_fit.csv', index=True)
        manager.save_df(pd.DataFrame(), 'threshold.csv', index=True)


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

################################################################################
# Code for plotting cohort behavior across animals
################################################################################
def plot_animal_perf(ax, animal_df, palette):
    date_start = animal_df.index.min()
    date_end = animal_df.index.max()
    df = animal_df.reindex(pd.date_range(date_start, date_end))

    for depth, color in palette.items():
        if depth in df:
            ax.plot(df[depth], marker='o', linestyle='-', color=color, label=str(depth))

    ax.axhline(0.5, ls=':', color='k')
    ax.axis(ymin=0, ymax=1)
    ax.set_ylabel('Performance\n(frac. correct)')
    add_weekends(ax, date_start, date_end)


def summarize_cohort(which, test_param, nogo_value, path=None):
    ds = dataset.Dataset(path=path)

    perf = getattr(ds, f'load_{which}_performance')()
    perf = perf.groupby(['animal_id', 'date', test_param])[['k', 'n']].sum()
    perf['p'] = perf.eval('k/n')

    n_animal = len(perf.index.unique('animal_id'))
    test_values = sorted(list(perf.index.unique(test_param)))
    test_values.remove(nogo_value)
    n_depth = max(len(test_values), 3)

    colors = getattr(palettable.scientific.sequential, f'Imola_{n_depth}')
    palette = {d: c for d, c in zip(test_values, colors.mpl_colors)}
    palette[0] = 'k'

    figure, axes = plt.subplots(n_animal, 1, figsize=(10, 2 * n_animal),
                                sharex=True, sharey=True, constrained_layout=True)

    for ax, (animal, animal_df) in zip(axes, perf.groupby('animal_id')):
        animal_df = animal_df.reset_index('animal_id', drop=True)
        animal_df = animal_df['p'].unstack(test_param)
        plot_animal_perf(ax, animal_df, palette)
        ax.set_title(animal)
        ax.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=3)

    plt.setp(axes[-1].get_xticklabels(), rotation=90)
    figure.savefig(ds.path / f'cohort-{which}.pdf')
