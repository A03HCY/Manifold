import matplotlib as mpl

def set_scientific_style():
    """Sets a scientific publication style for matplotlib."""
    pass

def plot_smoothed(ax, x, y, label, color, window=3):
    '''
    Plots original data with low alpha and smoothed data on top.
    '''
    # Original data
    ax.plot(x, y, alpha=0.3, color=color, linestyle='--')
    # Smoothed data
    y_smooth = y.rolling(window=window, min_periods=1).mean()
    ax.plot(x, y_smooth, label=label, color=color, linestyle='-', linewidth=1)
