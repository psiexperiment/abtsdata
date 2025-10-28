from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .dataset import Dataset
from .summarize_tinnitus_2AFC import palette


def plot_animal_perf(ax, animal_df):
    date_start = animal_df.index.min()
    date_end = animal_df.index.max()
    df = animal_df.reindex(pd.date_range(date_start, date_end))

    for ttype, color in palette.items():
        ax.plot(df[ttype], marker='o', linestyle='-', color=color, label=ttype)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.axhline(0.5, ls=':', color='k')
    ax.axis(ymin=0, ymax=1)
    ax.set_ylabel('Performance\n(frac. correct)')
    add_weekends(ax, date_start, date_end)


def add_weekends(ax, date_start, date_end):
    for date in pd.date_range(date_start, date_end):
        if date.day_of_week == 5:
            ax.axvspan(date - pd.Timedelta(days=0.5), date + pd.Timedelta(days=1.5), alpha=0.5)
            ax.text(date + pd.Timedelta(days=0.5), 0.75, 'weekend', transform=ax.get_xaxis_transform(), rotation=90, ha='center', va='center')


def summarize_cohort(path=None):
    ds = Dataset(path=path)

    df = ds.load_tinnitus_2AFC_performance()
    df_correct = df.groupby(['animal_id', 'date', 'trial_type'])[['size', 'sum']] \
        .sum() \
        .rename(columns={'size': 'n', 'sum': 'correct'}) \
        .eval('correct/n') \
        .unstack('trial_type')

    n_animals = len(df_correct.index.unique('animal_id'))

    figure, axes = plt.subplots(n_animals, 1,
                                figsize=(10, n_animals * 2),
                                sharex=True, sharey=True,
                                constrained_layout=True)

    for ax, (animal_id, animal_df) in zip(axes, df_correct.groupby('animal_id')):
        animal_df = animal_df.reset_index(['animal_id'], drop=True)
        plot_animal_perf(ax, animal_df)
        ax.set_title(animal_id)
    plt.setp(axes[-1].get_xticklabels(), rotation=90)
    figure.savefig(ds.path / 'cohort-performance.pdf')

    df = ds.load_tinnitus_2AFC_stats()
    df_stats = df.groupby(['animal_id', 'date']).agg({
        'n_trials': 'sum',
        'n_pellets': 'sum',
    })

    query = {
        'min_np_duration': 'context.parameters.trial.np_duration.expression',
        'to_duration': 'context.parameters.trial.to_duration.expression',
        'nbn_n': 'context.parameters."Tinnitus 2AFC".nbn_n.expression',
        'sam_n': 'context.parameters."Tinnitus 2AFC".sam_n.expression',
        'silence_n': 'context.parameters."Tinnitus 2AFC".silence_n.expression',
        'reward_rate': 'context.parameters."Tinnitus 2AFC".reward_rate.expression',
        'reward_silent': 'context.parameters."Tinnitus 2AFC".reward_silent.expression',
    }
    df = ds.load_raw_jmes('final.preferences', query=query) \
        .set_index(['animal_id', 'date', 'datetime']) \
        .sort_index()
    df_params = df.groupby(['animal_id', 'date'])[list(query)].last()
    df_params = df_stats.join(df_params)

    param_info = {
        'n_trials': '# of trials',
        'min_np_duration': 'Nose-poke dur. (s)',
        'to_duration': 'Timeout dur. (s)',
        'reward_rate': 'Reward Pr. (frac.)',
        'nbn_n': '# of NBN per set',
        'sam_n': '# of SAM per set',
        'silence_n': '# of silence per set',
    }

    n_params = len(param_info)

    figure, axes = plt.subplots(n_params, 1,
                                figsize=(10, n_params * 2),
                                sharex=True, sharey=False,
                                constrained_layout=True)

    animal_palette = {}
    for ax, (param, param_label) in zip(axes, param_info.items()):
        x = df_params[param].unstack('animal_id').astype('f')
        x = x.reindex(pd.date_range(x.index.min(), x.index.max()))
        for i, animal_id in enumerate(x):
            p, = ax.plot(
                x[animal_id], 'o-',
                color=animal_palette.get(animal_id, None),
                label=animal_id,
                lw=(n_animals + 1) - i,
                ms=(n_animals + 1) - i,
                zorder=i + 1,
                alpha=0.5,
            )
            animal_palette.setdefault(animal_id, p.get_color())
        ax.set_ylabel(param_label)
        add_weekends(ax, x.index.min(), x.index.max())

    axes[0].legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.setp(axes[-1].get_xticklabels(), rotation=90)
    figure.savefig(ds.path / 'cohort-params.pdf')


def main():
    import argparse
    parser = argparse.ArgumentParser('summarize-cohort-tinnitus-2AFC')
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    summarize_cohort(args.path)
