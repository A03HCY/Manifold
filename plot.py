import matplotlib as mpl

def set_scientific_style():
    '''
    Sets a scientific publication style for matplotlib with larger fonts.
    '''
    mpl.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 20
    })

def plot_smoothed(ax, x, y, label, color, window=3):
    '''
    Plots original data with low alpha and smoothed data on top.
    '''
    # Original data
    ax.plot(x, y, alpha=0.3, color=color, linestyle='--')
    # Smoothed data
    y_smooth = y.rolling(window=window, min_periods=1).mean()
    ax.plot(x, y_smooth, label=label, color=color, linestyle='-', linewidth=1)
