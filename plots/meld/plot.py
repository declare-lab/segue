from matplotlib import pyplot as plt
import csv
import itertools
import numpy as np
import pandas as pd
import scipy.stats as stats

GREEN = '#1f78b4'
PURPLE = "#D9791E"

def add_to_plot(x, y, c, ax):

    ln_x = np.log(x)
    z = np.polyfit(ln_x, y, 1)
    y_model = np.polyval(z, ln_x)   # modeling...
    p = np.poly1d(z)

    lnx_mean = np.mean(ln_x)
    y_mean = np.mean(y)
    n = x.size                        # number of samples
    m = 2                             # number of parameters
    dof = n - m                       # degrees of freedom
    t = stats.t.ppf(0.975, dof)
    residual = y - y_model

    std_error = (np.sum(residual**2) / dof)**.5   # Standard deviation of the error

    # to plot the adjusted model
    lnx_line = np.linspace(np.log(np.min(x)), np.log(np.max(x)), 100)
    y_line = np.polyval(z, lnx_line)
    ci = t * std_error * (1/n + (lnx_line - lnx_mean)**2 / np.sum((np.log(x) - lnx_mean)**2))**.5
    ax.fill_between(np.exp(lnx_line), y_line + ci, y_line - ci, color = f"{c}60", label = '_nolegend_', zorder=-100)

    plt.scatter(x, y, c=c, zorder=100)
    plt.plot(x, p(ln_x), '--', c=c)

def plot_ours_vs_baseline(x, ours, baseline, ours_full, baseline_full, name):
    plt.clf()
    fig = plt.figure()
    ax = fig.add_axes([.1, .1, .8, .8])
    ax.set_xscale('log', base=2)

    add_to_plot(x, baseline, PURPLE, ax)
    add_to_plot(x, ours, GREEN, ax)

    plt.legend(['w2v2', 'w2v2 trendline', 'SEGUE', 'SEGUE trendline'])
    plt.savefig(f'{name}.pdf',)# bbox_inches='tight')

def main():
    ours_data = pd.read_csv('ours-sent.csv')
    baseline_data = pd.read_csv('baseline-sent.csv')
    # with open('ours.csv') as file:
    #     reader = csv.reader(file)
    #     reader = itertools.islice(reader, 1, None)
    #     ours_data = list(ours_data)
    # with open('w2v2.csv') as file:
    #     reader = csv.reader(file)
    #     reader = itertools.islice(reader, 1, None)
    #     baseline_data = list(baseline_data)
    plot_ours_vs_baseline(
        ours_data['n'][3:], ours_data['score'][3:], baseline_data['score'][3:],
        np.mean(ours_data['score'][:3]), np.mean(baseline_data['score'][:3]),
        'sent',
    )

    ours_data = pd.read_csv('ours-emo.csv')
    baseline_data = pd.read_csv('baseline-emo.csv')
    plot_ours_vs_baseline(
        ours_data['n'][3:], ours_data['score'][3:], baseline_data['score'][3:],
        np.mean(ours_data['score'][:3]), np.mean(baseline_data['score'][:3]),
        'emo',
    )

if __name__ == '__main__':
    main()
