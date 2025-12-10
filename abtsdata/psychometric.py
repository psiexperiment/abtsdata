import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from scipy import stats
import xarray as xa


def fit_single_session(data, slope='negative'):
    x = data.index.values
    n = data['n'].values
    k = data['k'].values
    with pm.Model() as m:
        # This sets the midpoint of the function
        a = pm.Normal('alpha', 0, 10)
        #b = pm.Normal('beta', 0, 10)

        bt = pm.LogNormal('beta transformed', 1)
        if slope == 'negative':
            b = pm.Deterministic('beta', -bt)
        else:
            b = pm.Deterministic('beta', bt)

        gt = pm.Normal('gamma transformed', 0, 1)
        lt = pm.Normal('lambda transformed', 0, 1)

        # Gamma is FA rate, lambda is lapse (no response) rate.
        g = glogit(np.exp(gt), np.exp(lt))
        l = glogit(np.exp(lt), np.exp(gt))

        pm.Deterministic('gamma', g)
        pm.Deterministic('lambda', l)
        p = compute_psi(x, a, b, g, l)
        B = pm.Binomial('B', n=n, p=p, observed=k)
        return pm.sample(2000, nuts_sampler='numpyro', target_accept=0.8)


logit = lambda x: x/(1+x)
glogit = lambda x, y: x/(1+x+y)


def compute_psi(d, a, b, g, l):
    '''
    Standard psychometric function
    '''
    return g+(1-g-l)/(1+np.exp(-(d-a)/b))


def compute_psi_logit(d, a, b, g, l):
    ea = np.exp(a)
    eb = np.exp(b)
    eg = np.exp(g)
    el = np.exp(l)
    gi = glogit(eg, el)
    li = glogit(el, eg)
    ai = logit(ea)
    bi = logit(eb)
    return compute_psi(d, ai, bi, gi, li)


def fit_psychometric(summary, hdi_prob=0.9, ref_value=0):
    '''
    Fit psychometric function
    '''
    trace = fit_single_session(summary)

    # Create a sensible range for plotting the psychometric fit.
    x = summary.index.unique()
    x_lb, x_ub = x.min(), x.max()
    x_delta = (x_ub - x_lb) * 0.1
    x_lb -= x_delta
    x_ub += x_delta
    x_fit = np.linspace(x_lb, x_ub, 100)
    x_fit = xa.DataArray(x_fit, dims=['stm_depth'], coords={'stm_depth': x_fit})

    a = trace.posterior['alpha']
    b = trace.posterior['beta']
    g = trace.posterior['gamma']
    l = trace.posterior['lambda']

    # Broadcasting is intelligently handled because all coefficients are DataArrays.
    fit = compute_psi(x_fit, a, b, g, l)
    fit_d = xa.apply_ufunc(stats.norm.ppf, fit) - xa.apply_ufunc(stats.norm.ppf, g)
    fit_p_df = az.hdi(fit, hdi_prob=hdi_prob).to_dataframe()['x'].unstack('hdi')
    fit_p_df['mean'] = fit.mean(dim=('chain', 'draw')).to_dataframe()['x']
    fit_d_df = az.hdi(fit_d, hdi_prob=hdi_prob).to_dataframe()['x'].unstack('hdi')
    fit_d_df['mean'] = fit_d.mean(dim=('chain', 'draw')).to_dataframe()['x']
    fit_d_df = fit_d_df.rename(columns={'lower': 'lb', 'higher': 'ub'}).add_prefix('d_')
    fit_p_df = fit_p_df.rename(columns={'lower': 'lb', 'higher': 'ub'}).add_prefix('p_')
    fit_df = fit_d_df.join(fit_p_df)

    # Calculate threshold
    threshold = xa.apply_ufunc(
        lambda x, xp, fp: np.interp(x, xp[::-1], fp[::-1]),
        1,
        fit_d,
        fit_d.stm_depth,
        input_core_dims=[[], ['stm_depth'], ['stm_depth']],
        output_core_dims=[[]],
        exclude_dims={'stm_depth'},
        vectorize=True,
    )
    th_series = az.hdi(threshold, hdi_prob=hdi_prob).to_dataframe()['x'].rename({'lower': 'lb', 'higher': 'ub'})
    th_series['mean'] = np.mean(threshold.values)

    return {
        'trace': trace,
        'summary': summary,
        'fit': fit_df,
        'threshold': th_series,
    }


def plot_psychometric(summary, which, fit=None, threshold=None, color='k',
                      ax=None, show_ci=True, text_y=0.05, **kw):
    if ax is None:
        figure, ax = plt.subplots(1, 1, figsize=(4, 4), constrained_layout=True)
    else:
        figure = ax.figure

    if which == 'p':
        ax.scatter(summary.index.values, summary['p'] * 100, summary['n'], color=color)
        if fit is not None:
            ax.plot(fit.index.values, fit['p_mean'] * 100, '-', color=color)
            if show_ci:
                ax.fill_between(fit.index.values, fit['p_lb'] * 100, fit['p_ub'] * 100, color=color, alpha=0.2)
        ax.set_ylabel('Percent Yes (%)')
        ax.axis(ymin=-5, ymax=105)
    elif which == 'd':
        ax.scatter(summary.index.values, summary['d'], summary['n'], color=color)
        if fit is not None:
            ax.plot(fit.index.values, fit['d_mean'], '-', color=color)
            ax.axhline(1, ls=':', color=color)
        if threshold is not None:
            m, lb, ub = threshold[['mean', 'lb', 'ub']]
            ax.axvline(m, ls=':', color=color)
            if show_ci:
                ax.fill_between(fit.index.values, fit['d_lb'], fit['d_ub'], color=color, alpha=0.2)
                ax.axvspan(lb, ub, color=color, alpha=0.2)
            ax.text(0.05, text_y, f'{m:.1f} ({lb:.1f} to {ub:.1f})',
                    fontsize='x-small', transform=ax.transAxes, color=color)
        ax.set_ylabel("$d'$")

    ax.set_xlabel('STM depth (dB)')
    return figure, ax
