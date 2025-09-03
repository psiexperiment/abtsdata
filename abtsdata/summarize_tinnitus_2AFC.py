import matplotlib.pyplot as plt
import numpy as np
from palettable.tableau import BlueRed_6

from psidata.api import Recording
from psidata.util import add_default_options, DatasetManager, process_files


ttypes = ['NBN', 'SAM', 'silence']
palette = {t: c for t, c in zip(ttypes, BlueRed_6.mpl_colors)}


expected_suffixes = [
    'performance.pdf',
]


def process_file(filename, manager):
    with manager.create_cb() as cb:
        fh = Recording(filename)

    figure, axes = plt.subplot_mosaic(
        [['correct', 'correct', 'correct'],
        ['tpm', 'tpm', 'tpm'],
        ['rt', 'cat_perf', 'nbn_perf']],
        figsize=(10, 8), layout='constrained')
    axes['correct'].sharex(axes['tpm'])

    # Correct vs. incorrect by trial
    tl = fh.trial_log.copy()
    tl['score'] = tl['correct'].astype('f') + np.random.uniform(-0.1, 0.1, size=len(tl))
    m = tl['response'] == 'no_response'
    tl.loc[m, 'score'] = np.random.uniform(-0.1, 0.1, size=m.sum()) - 1

    for ttype, color in palette.items():
        x = tl.query('trial_type == @ttype')
        axes['correct'].plot(x.index, x['score'], 'o', mec='w', mew=1, label=ttype, color=color)
    axes['correct'].axis(ymin=-1.2, ymax=1.2)
    axes['correct'].yaxis.set_ticks([-1, 0, 1])
    axes['correct'].yaxis.set_ticklabels(['No resp.', 'Incorrect', 'Correct'])
    axes['correct'].legend(bbox_to_anchor=(1, 1), loc='upper left')
    axes['correct'].tick_params(labelbottom=False)

    # Trials per minute
    tpm = 60 / tl['trial_start'].diff()
    axes['tpm'].plot(tpm, label='Per trial')
    axes['tpm'].plot(tpm.rolling(5).mean(), label='Moving average')
    axes['tpm'].set_xlabel('Trial Number')
    axes['tpm'].set_ylabel('Trials per minute')
    axes['tpm'].legend(bbox_to_anchor=(1, 1), loc='upper left')

    # Plot reaction time
    sns.kdeplot(fh.trial_log, x='response_time', hue='trial_type',
                palette=palette, common_norm=True, clip=(0, 8), ax=axes['rt'])
    l = axes['rt'].get_legend()
    l.set_title(None)
    axes['rt'].set_xlabel('Response Time (sec)\n(trial start to hopper)')

    # Plot category performance
    mean_correct = fh.trial_log.groupby('trial_type')['correct'].mean()
    for i, (ttype, color) in enumerate(palette.items()):
        y = mean_correct.loc[ttype]
        axes['cat_perf'].bar(i, y, color=color)
    axes['cat_perf'].xaxis.set_ticks([0, 1, 2])
    axes['cat_perf'].xaxis.set_ticklabels(ttypes)
    axes['cat_perf'].set_ylabel('Fraction correct')
    axes['cat_perf'].axis(ymin=0, ymax=1)
    axes['cat_perf'].axhline(0.5, ls=':', color='k')

    tl = fh.trial_log.copy()
    tl['nbn_frequency'] = tl['nbn_frequency'].apply(lambda x: round(x / 1e3, 1))
    nbn_mean = tl.query('trial_type == "NBN"').groupby(['nbn_frequency'])['correct'].mean()
    x = range(len(nbn_mean))
    axes['nbn_perf'].bar(x, nbn_mean, color=palette['NBN'])
    axes['nbn_perf'].xaxis.set_ticks(x)
    axes['nbn_perf'].xaxis.set_ticklabels(nbn_mean.index, rotation=90)
    axes['nbn_perf'].set_xlabel('NBN frequency (kHz)') 
    axes['nbn_perf'].set_ylabel('Fraction correct')
    axes['nbn_perf'].axis(ymin=0, ymax=1)
    axes['nbn_perf'].axhline(0.5, ls=':', color='k')

    manager.save_fig(figure, 'performance.pdf')


def main():
    import argparse
    parser = argparse.ArgumentParser('Summarize tinnitus 2AFC data in folder')
    add_default_options(parser)
    args = vars(parser.parse_args())
    process_files('**/*tinnitus-2AFC*', process_file,
                  expected_suffixes=expected_suffixes, **args)
