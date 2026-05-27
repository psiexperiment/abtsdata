import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from scipy import stats
import xarray as xa


def fit_single_session(data, sigmoid, core, slope, alpha, beta,
                       target_accept=0.8):
    x = data.index.values
    n = data['n'].values
    k = data['k'].values

    with pm.Model() as m:
        dist = alpha.pop('dist')
        a = getattr(pm, dist)('alpha', **alpha)
        dist = beta.pop('dist')
        bt = getattr(pm, dist)('beta_transformed', **beta)

        if slope == 'negative':
            b = pm.Deterministic('beta', -bt)
        else:
            b = pm.Deterministic('beta', bt)


        gt = pm.Normal('gamma transformed', -1, 1)
        lt = pm.Normal('lambda transformed', -2.5, 1)

        # Gamma is FA rate, lambda is lapse (no response) rate.
        g = glogit(pm.math.exp(gt), pm.math.exp(lt))
        l = glogit(pm.math.exp(lt), pm.math.exp(gt))

        pm.Deterministic('gamma', g)
        pm.Deterministic('lambda', l)
        p = compute_psi(x, a, b, g, l, sigmoid, core, pymc=True)
        B = pm.Binomial('B', n=n, p=p, observed=k)
        return pm.sample(2000, nuts_sampler='numpyro',
                         target_accept=target_accept)


logit = lambda x: x/(1+x)
glogit = lambda x, y: x/(1+x+y)


def compute_psi(x, a, b, g, l, sigmoid='logistic', core='ab', pymc=False):
    '''
    Standard psychometric function
    '''
    exp = pm.math.exp if pymc else np.exp

    if core == 'ab':
        c = (x - a) / b
    elif core == 'poly':
        c = (x / a) ** b

    if sigmoid == 'logistic':
        s = 1 / (1 + exp(-c))
    elif sigmoid == 'exponential':
        s = 1 - exp(-c)
    elif sigmoid == 'gumbel_r':
        s = exp(-exp(-c))
    else:
        raise ValueError('Unrecognized psychometric function')

    return g + (1 - g - l) * s


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


def fit_psychometric(summary, hdi_prob=0.9, sigmoid='logistic', core='ab',
                     alpha=None, beta=None, slope='positive',
                     target_accept=0.8):
    '''
    Fit psychometric function
    '''
    trace = fit_single_session(summary, sigmoid=sigmoid, core=core,
                               alpha=alpha, beta=beta, slope=slope,
                               target_accept=target_accept)
    param = summary.index.name

    # Create a sensible range for plotting the psychometric fit.
    x = summary.index.unique()
    x_lb, x_ub = x.min(), x.max()
    x_delta = (x_ub - x_lb) * 0.1
    x_lb -= x_delta
    x_ub += x_delta
    x_fit = np.linspace(x_lb, x_ub, 100)
    x_fit = xa.DataArray(x_fit, dims=[param], coords={param: x_fit})

    a = trace.posterior['alpha']
    b = trace.posterior['beta']
    g = trace.posterior['gamma']
    l = trace.posterior['lambda']

    # Broadcasting is intelligently handled because all coefficients are DataArrays.
    fit = compute_psi(x_fit, a, b, g, l, sigmoid, core)
    fit_d = xa.apply_ufunc(stats.norm.ppf, fit) - xa.apply_ufunc(stats.norm.ppf, g)
    fit_p_df = az.hdi(fit, hdi_prob=hdi_prob).to_dataframe()['x'].unstack('hdi')
    fit_p_df['mean'] = fit.mean(dim=('chain', 'draw')).to_dataframe()['x']
    fit_d_df = az.hdi(fit_d, hdi_prob=hdi_prob).to_dataframe()['x'].unstack('hdi')
    fit_d_df['mean'] = fit_d.mean(dim=('chain', 'draw')).to_dataframe()['x']
    fit_d_df = fit_d_df.rename(columns={'lower': 'lb', 'higher': 'ub'}).add_prefix('d_')
    fit_p_df = fit_p_df.rename(columns={'lower': 'lb', 'higher': 'ub'}).add_prefix('p_')
    fit_df = fit_d_df.join(fit_p_df)

    if slope == 'positive':
        fn = lambda x, xp, fp: np.interp(x, xp, fp)
    else:
        fn = lambda x, xp, fp: np.interp(x, xp[::-1], fp[::-1])

    # Calculate threshold
    threshold = xa.apply_ufunc(
        fn,
        1,
        fit_d,
        fit_d[param],
        input_core_dims=[[], [param], [param]],
        output_core_dims=[[]],
        exclude_dims={param},
        vectorize=True,
    )
    th_series = az.hdi(threshold, hdi_prob=hdi_prob).to_dataframe()['x'].rename({'lower': 'lb', 'higher': 'ub'})
    th_series['mean'] = np.mean(threshold.values)

    fa = g.mean(dim=('chain', 'draw'))
    summary = summary.copy()
    summary['d_prime'] = stats.norm.ppf(summary['p']) - stats.norm.ppf(fa)

    return {
        'trace': trace,
        'summary': summary,
        'fit': fit_df,
        'threshold': th_series,
    }


def plot_psychometric(summary, which, fit=None, threshold=None, color='k',
                      ax=None, show_ci=True, text_y=0.05, x_label='',
                      x_scale='linear', **kw):
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

    ax.set_xlabel(x_label)
    return figure, ax
