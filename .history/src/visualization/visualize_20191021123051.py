# Data handling and processing
import pandas as pd
import numpy as np

# Data visualisation & images
import matplotlib.pyplot as plt
import seaborn as sns


def importance_plotting(data, x, y, palette, title):
    """
    Importance plotting of features came from the Kaggle Kernel of Joshua Reed.  
    He used this plot to identify important features. 

    https://www.kaggle.com/josh24990/simple-end-to-end-ml-workflow-top-5-score
    
    Arguments:
        data {[type]} -- [description]
        x {[type]} -- [description]
        y {[type]} -- [description]
        palette {[type]} -- [description]
        title {[type]} -- [description]
    """
    sns.set(style="whitegrid")
    ft = sns.PairGrid(data, y_vars=y, x_vars=x, size=5, aspect=1.5)
    ft.map(sns.stripplot, orient='h', palette=palette, edgecolor="black", size=15)
    
    for ax, title in zip(ft.axes.flat, titles):
    # Set a different title for each axes
        ax.set(title=title)
    # Make the grid horizontal instead of vertical
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)
    plt.show()