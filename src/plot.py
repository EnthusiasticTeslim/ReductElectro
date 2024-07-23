import sys
sys.dont_write_bytecode = True

from .utils import to_np
import numpy as np
import seaborn as sns

# Scatter plot function
def plot_scatter(ax, data, row, col, column_names, torch=True):
    if torch:
        ax[row, col].scatter(to_np(data['y_train'][:, col]), to_np(data['output_train'][:, col]), color='blue', label='Train')
        ax[row, col].scatter(to_np(data['y_test'][:, col]), to_np(data['output_test'][:, col]), color='green', label='Test')
    else:
        ax[row, col].scatter(data['y_train'][:, col], data['output_train'][:, col], color='blue', label='Train')
        ax[row, col].scatter(data['y_test'][:, col], data['output_test'][:, col], color='green', label='Test')

    ax[row, col].plot([0, 1], [0, 1], transform=ax[row, col].transAxes, color='red', linestyle='--')
    ax[row, col].set_title(column_names[col])
    ax[row, col].set_xlabel(r'$\rm True\ FE$')
    if row == 0 and col == 0:
        ax[row, col].legend()
        ax[row, col].set_ylabel(r'$\rm Predicted\ FE$')
        

# KDE plot function
def plot_kde(ax, data, row, col, torch=True):
    if torch:
        y_data = np.concatenate([to_np(data['y_train'][:, col]), to_np(data['y_test'][:, col])])
        output_data = np.concatenate([to_np(data['output_train'][:, col]), to_np(data['output_test'][:, col])])
    else:
        y_data = np.concatenate([data['y_train'][:, col], data['y_test'][:, col]])
        output_data = np.concatenate([data['output_train'][:, col], data['output_test'][:, col]])
        
    error = np.abs(y_data - output_data)
    _, _, hist = ax[row, col].hist(error, density=True, alpha=0.5)
    sns.kdeplot(error, ax=ax[row, col], color=hist.patches[0].get_facecolor(), lw=2)
    ax[row, col].set_xlabel(r'$\rm Error$')
    
    # if row == 1 and col == 0:
    #     ax[row, col].set_ylabel(r'$\rm Density$')