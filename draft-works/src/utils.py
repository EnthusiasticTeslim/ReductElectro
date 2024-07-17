import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pymatgen.core as pmg

def create_structure(Sn_percent):
    # create the structure
    if Sn_percent <= 1:
        base = f'Cu{1-Sn_percent}Sn{Sn_percent}'
        comp = pmg.Composition(base)
    else:
        raise ValueError('Sn % must be <= 1')
    return comp



def load_data(data_path):
    # the data
    data = pd.read_excel(data_path)
    data = data.drop(columns=['S/N'])

    # normalize the data in target columns by 100
    features_col = list(data.columns[:4])
    target_col = list(data.columns[4:])

    data[target_col] = data[target_col] / 100 # normalize the target data by 100
    data[features_col[2]] = data[features_col[2]] / 100 # normalize the Sn% by 100
    print('Features: ', features_col)
    print('Target: ', target_col)

    data['Cu %'] = 1 - data['Sn %']
    data['weight'] = data['Sn %'].apply(create_structure).apply(lambda x: x.weight)

    # normalize the data in features columns to range [0, 1]
    features_col += ['Cu %', 'weight']
    print(f'New features: {features_col}')
    minX = data[features_col].min()
    maxX = data[features_col].max()

    data[features_col] = (data[features_col] - minX) / (maxX - minX)

    return data, features_col, target_col


def plot_heat_map(
                    data: pd.DataFrame = None, mask: bool = False, compute_corr: bool = True,
                    fig_size = (10, 5), save_fig: bool = False, name: str = 'general'
                    ):
    ''' Plot the heatmap of the correlation matrix of the data  '''
        
    fig, ax = plt.subplots(1, figsize=fig_size, facecolor='white')

    # Create the heatmap for the original data
    if compute_corr:
        corr = data.corr(method='pearson')
    else:
        corr = data
    if mask is False:
        sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f', linewidths=.5, ax=ax, cbar=False)
    else: # mask the diagonal
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f', linewidths=.5, ax=ax, cbar=False, mask=mask)
        
    # Show the plot
    plt.show()
    if save_fig:
        fig.savefig(f'./reports/heatmap_{name}.png', dpi=200)