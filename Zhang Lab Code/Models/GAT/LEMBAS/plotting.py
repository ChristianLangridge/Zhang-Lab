"""
Helper functions for data visualization. 
"""
import math

import pandas as pd
import numpy as np
import plotnine as p9

def shade_plot(X: np.array, Y: np.array, sigma: np.array, x_label: str, y_label: str, 
              width: int = 5, height: int = 3):
    """_summary_

    Parameters
    ----------
    X : np.array
        x axis values
    Y : np.array
        y axis values
    sigma : np.array
        standard deviation of y axis values
    x_label : str
        x axis label 
    y_label : str
        y axis label

    Returns
    -------
    plot : plotnine.ggplot.ggplot
        _description_
    """
    data = pd.DataFrame(data = {
        x_label: X, y_label: Y, 'sigma': sigma
    })
    data['sigma_min'] = data[y_label] - data.sigma
    data['sigma_max'] = data[y_label] + data.sigma

    plot = (
        p9.ggplot(data, p9.aes(x=x_label)) +
        p9.geom_line(p9.aes(y=y_label), color = '#1E90FF') +
        p9.geom_ribbon(p9.aes(y = y_label, ymin = 'sigma_min', ymax = 'sigma_max'), alpha = 0.2) +
        p9.xlim([0, data.shape[0]]) +
        # p9.ylim(10**min_log, round(data[y_label].max(), 1)) +
        # p9.scale_y_log10() + 
        p9.theme_bw() + 
        p9.theme(figure_size=(width, height))
    )
    return plot