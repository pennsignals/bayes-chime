from os import getcwd, path
import json

from configargparse import ArgParser
from scipy import stats as sps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# plot of logistic curves
def logistic(L, k, x0, x):
    return L/(1+np.exp(-k*(x-x0)))

# plot of chains
def plt_predictive(df, first_day, census_ts, hosp_capacity, vent_capacity,
                   figdir, as_of_days_ago, howfar=200, y_max=None, prefix=''):
    # predictive plot
    file_howfar = howfar
    arrs = np.stack([df.arr.iloc[i] for i in range(df.shape[0])])
    arrq = np.quantile(arrs, axis = 0, q = [.025, .25, .5, .75, .975])
    howfar = len(census_ts.hosp) + howfar
    howfar = np.min([howfar, arrs.shape[1]])

    dates = pd.date_range(f'{first_day}',
                          periods=howfar, freq='d')
    fig, ax = plt.subplots(figsize=(16, 10), ncols=2, nrows=2, sharex=True)
    # hosp
    axx = ax[0,0]
    if y_max:
        axx.set_ylim((0, y_max))
    axx.plot_date(dates, arrq[2,:howfar,3], '-', label = 'posterior median')
    axx.set_ylabel(f'COVID-19 Hospital census', fontsize=12, fontweight='bold')
    axx.fill_between(x = dates,
                     y1 = arrq[0,:howfar,3],
                     y2 = arrq[4,:howfar,3],
                     label = '95% Credible Region',
                     alpha = .1,
                     lw = 2,
                     edgecolor = "k")
    axx.fill_between(x = dates,
                     y1 = arrq[1,:howfar,3],
                     y2 = arrq[3,:howfar,3],
                     label = '50% Credible Region',
                     alpha = .1,
                     lw = 2,
                     edgecolor = "k")
    axx.plot_date(dates[:census_ts.hosp.shape[0]], census_ts.hosp, '-',
                  color = "red",
                  label = "observed")
    axx.axhline(y=hosp_capacity, color='k', ls='--', label = "hospital capacity")
    axx.axvline(x= dates.values[census_ts.hosp.shape[0]-as_of_days_ago],
      color='grey', ls='--', label = "Last Datapoint Used")
    
    axx.legend()
    axx.grid(True)

    axx = ax[0,1]
    if y_max:
        axx.set_ylim((0, y_max))
    axx.plot_date(dates, arrq[2,:howfar,5], '-', label = 'posterior median')
    axx.set_ylabel(f'COVID-19 Vent census', fontsize=12, fontweight='bold')
    axx.fill_between(x = dates,
                     y1 = arrq[0,:howfar,5],
                     y2 = arrq[4,:howfar,5],
                     label = '95% Credible Region',
                     alpha = .1,
                     lw = 2,
                     edgecolor = "k",)
    axx.fill_between(x = dates,
                     y1 = arrq[1,:howfar,5],
                     y2 = arrq[3,:howfar,5],
                     label = '50% Credible Region',
                     alpha = .1,
                     lw = 2,
                     edgecolor = "k")
    axx.axhline(y=vent_capacity, color='k', ls='--', label = "vent capacity")
    axx.plot_date(dates[:census_ts.vent.shape[0]], census_ts.vent, '-',
                  color = "red",
                  label = "observed")
    axx.axvline(x= dates.values[census_ts.hosp.shape[0]-as_of_days_ago],
      color='grey', ls='--', label = "Last Datapoint Used")
    axx.legend()
    axx.grid(True)

    # Admits
    axx = ax[1,0]
    axx.plot_date(dates,arrq[2,:howfar,0], '-', label = 'posterior median')
    axx.set_ylabel(f'COVID-19 Hospital Admits', fontsize=12, fontweight='bold')
    axx.fill_between(x = dates,
                     y1 = arrq[0,:howfar,0],
                     y2 = arrq[4,:howfar,0],
                     label = '95% Credible Region',
                     alpha = .1,
                     lw = 2,
                     edgecolor = "k")
    axx.fill_between(x = dates,
                     y1 = arrq[1,:howfar,0],
                     y2 = arrq[3,:howfar,0],
                     label = '50% Credible Region',
                     alpha = .1,
                     lw = 2,
                     edgecolor = "k")
    axx.legend()
    axx.grid(True)

    axx = ax[1,1]
    axx.plot_date(dates, arrq[2,:howfar,2], '-', label = 'posterior median')
    axx.set_ylabel(f'COVID-19 Vent Admits', fontsize=12, fontweight='bold')
    axx.fill_between(x = dates,
                     y1 = arrq[0,:howfar,2],
                     y2 = arrq[4,:howfar,2],
                     label = '95% Credible Region',
                     alpha = .1,
                     lw = 2,
                     edgecolor = "k",)
    axx.fill_between(x = dates,
                     y1 = arrq[1,:howfar,2],
                     y2 = arrq[3,:howfar,2],
                     label = '50% Credible Region',
                     alpha = .1,
                     lw = 2,
                     edgecolor = "k")
    axx.legend()
    axx.grid(True)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path.join(f"{figdir}", f"{prefix}forecast_{file_howfar}_day.pdf"))


def mk_projection_tables(df, first_day, outdir):
    # predictive plot
    arrs = np.stack([df.arr.iloc[i] for i in range(df.shape[0])])
    arrq = np.quantile(arrs, axis = 0, q = [.05, .25, .5, .75, .95])
    column_postfix = ['5%', '25%', 'Median', '75%', '%95']

    summary_df_hsp = pd.DataFrame(data=arrq[:,:,3].T,
                                  columns=[f'Hospitalized Census {pf}' for pf in column_postfix])
    summary_df_vent = pd.DataFrame(data=arrq[:,:,5].T,
                                   columns=[f'Vent Census {pf}' for pf in column_postfix])


    summary_df_hsp_admits = pd.DataFrame(data=arrq[:,:,0].T.astype(int),
                                         columns=[f'Hospitalized Admits {pf}' for pf in column_postfix])
    summary_df_vent_admits = pd.DataFrame(data=arrq[:,:,2].T.astype(int),
                                          columns=[f'Vent Admits {pf}' for pf in column_postfix])

    date_df = pd.DataFrame(data=pd.date_range(f'{first_day}',
                                              periods=summary_df_hsp.shape[0], freq='d'),
                           columns = ['date'])

    summary_df = pd.concat([date_df,
                            summary_df_hsp,
                            summary_df_vent,
                            summary_df_hsp_admits,
                            summary_df_vent_admits], 1)
    summary_df.to_csv(path.join(f"{outdir}", "forecast.csv"), index=False)

def read_inputs(paramdir):
    with open(path.join(paramdir, "args.json"), "r") as f:
        args = json.load(f)
    census_ts = pd.read_csv(path.join(paramdir, "census_ts.csv"))
    params = pd.read_csv(path.join(paramdir, "params.csv"))
    return census_ts, params, args

def main():
    p = ArgParser()
    p.add("-c", "--my-config", is_config_file=True, help="config file path")
    p.add("-o", "--out", help="output directory")
    p.add(
        "-a",
        "--as_of",
        default=0,
        help="number of days in the past to project from",
        type=int,
    )
    p.add("-y", "--y_max", help="max y-scale for the census graph", type=int)
    p.add(
        "-d",
        "--n_days",
        help="make a census/admits plot out to n_days",
        type=int,
        action="append",
    )
    p.add("-P", "--prefix", help="prefix for filenames")
    options = p.parse_args()

    prefix = ''
    if options.prefix is not None:
        prefix = f"{options.prefix}_"

    n_days = [30, 90, 180]
    if options.n_days:
        n_days = options.n_days

    dir = options.out
    if not path.isdir(dir):
        dir = path.join(f"{getcwd()}", "output", options.out)
    paramdir = path.join(dir, "parameters")
    outdir = path.join(dir, 'output')
    figdir = path.join(dir, 'figures')

    census_ts, params, args = read_inputs(paramdir)
    first_day = census_ts['date'].values[0]

    # TODO: This needs to be configurable based on the time period specificed
    as_of_days_ago = args['as_of']
    nobs = census_ts.shape[0] - as_of_days_ago

    # define capacity
    vent_capacity = float(params.base.loc[params.param == 'vent_capacity'])
    hosp_capacity = float(params.base.loc[params.param == 'hosp_capacity'])

    # Chains
    df = pd.read_json(path.join(f'{outdir}', 'chains.json.bz2'), orient='records', lines=True)
    print(f"READ chains file: {df.shape[0]} total iterations")
    # remove burn-in
    # Make 1000 configurable
    df = df.loc[(df.iter>1000)] #& (~df.chain.isin([1, 12]))]



    qlist = []
    for day in range(census_ts.shape[0]):
        ldist = logistic(df.logistic_L,
                          df.logistic_k,
                          df.logistic_x0 - df.offset.astype(int),
                          day)
        qlist.append(np.quantile(ldist, [.05, .5, .95]))


    # logistic SD plot
    qmat = np.vstack(qlist)
    fig = plt.figure()

    plt.plot(list(range(census_ts.shape[0])), 1-qmat[:,1])
    plt.fill_between(x = list(range(census_ts.shape[0]))
                     ,y1=1-qmat[:,0]
                     ,y2 = 1-qmat[:,2]
                     ,alpha=.3
                     ,lw=2
                     ,edgecolor='k'
                   )
    plt.ylabel(f'Relative (effective) social contact')
    plt.xlabel(f'Days since {first_day}')
    plt.ylim(0,1)
    fig.savefig(path.join(f"{figdir}", f"{prefix}effective_soc_dist.pdf"))

    for howfar in n_days:
        plt_predictive(
            df, first_day, census_ts, hosp_capacity, vent_capacity, figdir,
            as_of_days_ago, howfar=howfar, prefix=prefix, y_max=options.y_max
        )

    mk_projection_tables(df, first_day, outdir)

    toplot = df[['beta',
                 'hosp_prop',
                 'ICU_prop',
                 'vent_prop',
                 'hosp_LOS',
                 'ICU_LOS',
                 'vent_LOS',
                 'incubation_days',
                 'recovery_days',
                 'logistic_k',
                 'logistic_x0',
                 'logistic_L']]
    #toplot.days_until_overacpacity[toplot.days_until_overacpacity == -9999] = np.nan

    pspace = np.linspace(.001, .999, 1000)

    fig, ax = plt.subplots(figsize=(8, 40), ncols=1, nrows=len(toplot.columns))
    for i in range(len(toplot.columns)):
        cname = toplot.columns[i]
        if params.loc[params.param == cname, 'distribution'].iloc[0] == 'gamma':
            x = sps.gamma.ppf(pspace, params.loc[params.param == cname, 'p1'], 0, params.loc[params.param == cname, 'p2'])
            y = sps.gamma.pdf(x, params.loc[params.param == cname, 'p1'], 0, params.loc[params.param == cname, 'p2'])
        elif params.loc[params.param == cname, 'distribution'].iloc[0] == 'beta':
            x = sps.beta.ppf(pspace, params.loc[params.param == cname, 'p1'], params.loc[params.param == cname, 'p2'])
            y = sps.beta.pdf(x, params.loc[params.param == cname, 'p1'], params.loc[params.param == cname, 'p2'])
        ax[i].plot(x, y, label = "prior")
        ax[i].hist(toplot[cname], density = True, label = "posterior", bins=30)
        ax[i].set_xlabel(params.loc[params.param == cname, 'description'].iloc[0])
        ax[i].legend()
    plt.tight_layout()
    fig.savefig(path.join(f'{figdir}', f'{prefix}marginal_posteriors_v2.pdf'))

if __name__ == '__main__':
    main()
