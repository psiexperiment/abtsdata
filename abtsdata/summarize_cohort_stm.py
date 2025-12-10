from pathlib import Path
import matplotlib.pyplot as plt
import palettable.scientific.sequential
import pandas as pd

from abtsdata import dataset
from abtsdata.plot import add_weekends


def plot_animal_perf(ax, animal_df, palette):
    date_start = animal_df.index.min()
    date_end = animal_df.index.max()
    df = animal_df.reindex(pd.date_range(date_start, date_end))

    for depth, color in palette.items():
        if depth in df:
            ax.plot(df[depth], marker='o', linestyle='-', color=color, label=str(depth))

    #ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.axhline(0.5, ls=':', color='k')
    ax.axis(ymin=0, ymax=1)
    ax.set_ylabel('Performance\n(frac. correct)')
    add_weekends(ax, date_start, date_end)


def summarize_cohort(path=None, which='gonogo'):
    ds = dataset.Dataset(path=path)

    perf = getattr(ds, f'load_modulation_{which}_performance')()
    perf = perf.groupby(['animal_id', 'date', 'stm_depth'])[['k', 'n']].sum()
    perf['p'] = perf.eval('k/n')

    n_animal = len(perf.index.unique('animal_id'))
    stm_depth = sorted(list(perf.index.unique('stm_depth')))
    stm_depth.remove(0)
    n_depth = max(len(stm_depth), 3)

    colors = getattr(palettable.scientific.sequential, f'Imola_{n_depth}')
    palette = {d: c for d, c in zip(stm_depth, colors.mpl_colors)}
    palette[0] = 'k'

    figure, axes = plt.subplots(n_animal, 1, figsize=(10, 2 * n_animal),
                                sharex=True, sharey=True, constrained_layout=True)

    for ax, (animal, animal_df) in zip(axes, perf.groupby('animal_id')):
        animal_df = animal_df.reset_index('animal_id', drop=True)
        animal_df = animal_df['p'].unstack('stm_depth')
        plot_animal_perf(ax, animal_df, palette)
        ax.set_title(animal)
        ax.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=3)

    plt.setp(axes[-1].get_xticklabels(), rotation=90)
    figure.savefig(ds.path / f'cohort-stm-{which}.pdf')


def main_gonogo():
    import argparse
    parser = argparse.ArgumentParser('summarize-cohort-modulation-gonogo')
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    summarize_cohort(args.path, 'gonogo')


def main_2AFC():
    import argparse
    parser = argparse.ArgumentParser('summarize-cohort-modulation-2AFC')
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    summarize_cohort(args.path, '2AFC')
