import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import shap


class Visualization:
    def __init__(self):
        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
        plt.rcParams['font.size'] = 13
        plt.rcParams['figure.dpi'] = 200

    def disp_shap_bar(self, shap_values, data, names, tag='H_2', color='red', path='str', save=False):
        # plot bar
        shap.summary_plot(shap_values, data, plot_type="bar", show=False, color=color, feature_names=names)
 
        fig = plt.gcf().set_size_inches((6, 4))
 
        plt.xlabel(fr'$ \rm Feature\ importance \ ({tag})$')

        if save:
            plt.savefig(path)
 
        plt.show()

    def disp_shap_bee(self, shap_values, data, names, tag='H_2', color='red', path='str', save=False):
        # plot bar
        shap.summary_plot(shap_values, data, show=False, color=color, feature_names=names)
 
        fig1 = plt.gcf().set_size_inches((6, 4))
 
        plt.xlabel(fr'$ \rm Feature\ importance \ ({tag})$')

        if save:
            plt.savefig(path)
 
        plt.show()

    
    def disp_shap_bee(self, shap_values, data, names, tag='H_2', color='red', path='str', save=False):
        # plot bar
        shap.summary_plot(shap_values, data, show=False, color=color, feature_names=names)
 
        fig1 = plt.gcf().set_size_inches((6, 4))
 
        plt.xlabel(fr'$ \rm Feature\ importance \ ({tag})$')

        if save:
            plt.savefig(path)
 
        plt.show()

    def show_crossplot(self, result, save=False, add_bar=False, path='str', label='H_2', figsize=(5, 4)):
        ''' Plot the crossplot of the results '''
        y_train = result['train']['y']
        y_train_pred_ = result['train']['y_pred']
        y_train_pred_sttd = result['train']['y_pred_sttd']

        # upper_train = result['train']['upper']
        # lower_train = result['train']['lower']

        y_test = result['test']['y']
        y_test_pred_ = result['test']['y_pred']
        y_test_pred_sttd = result['test']['y_pred_sttd']
        # upper_test = result['test']['upper']
        # lower_test = result['test']['lower']

        y = np.concatenate([y_train, y_test])
    
        # plot the results
        plt.figure(figsize=figsize) # 5,4
        if  add_bar: # with 1 standard deviation
            plt.errorbar(y_train, y_train_pred_, yerr=y_train_pred_sttd, fmt='o', color='red', label=r'$\rm Train$')
            plt.errorbar(y_test, y_test_pred_, yerr=y_test_pred_sttd, fmt='o', color='blue', label=r'$\rm Test$')
        else:
            plt.scatter(y_train, y_train_pred_, color='red', label=r'$\rm Train$')
            plt.scatter(y_test, y_test_pred_, color='blue', label=r'$\rm Test$')
        
        plt.plot([y.min() - 0.05, y.max() + 0.05], [y.min() - 0.05, y.max() + 0.05], color='black', linestyle='--')
        plt.xlabel(fr'$\rm Experiment \ {label} \ FE$', fontsize=10)
        plt.ylabel(fr'$\rm Predicted \ {label} \ FE$', fontsize=10)
        plt.legend(fontsize=8)
        # # save the plot and model
        if save:
            # save image
            plt.savefig(path) # os.path.join(args.output_path, f'gp_model_id{args.label_index}_seed{args.seed}.png')
        plt.show()


def plot_data(data, metric, metric_name='MAE', name='CO', pred_name='CO_pred', title='CO', figsize=(6, 3)):

    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].scatter(data[f'{name}'], data[f'{pred_name}_1'], label=r'$\rm 1$', marker='d', color='black')
    ax[0].scatter(data[f'{name}'], data[f'{pred_name}_2'], label=r'$\rm 2$', marker='o', color='red')
    ax[0].scatter(data[f'{name}'], data[f'{pred_name}_3'], label=r'$\rm 3$', marker='s', color='blue')
    ax[0].scatter(data[f'{name}'], data[f'{pred_name}_4'], label=r'$\rm 4$', marker='^', color='green')
    ax[0].scatter(data[f'{name}'], data[f'{pred_name}_5'], label=r'$\rm 5$', marker='v', color='purple')
    ax[0].scatter(data[f'{name}'], data[f'{pred_name}_6'], label=r'$\rm 6$', marker='x', color='orange')

    diff = data[f'{name}'].max() - data[f'{name}'].min()
    ax[0].plot([0, data[f'{name}'].max() + 0.1*diff], [0, data[f'{name}'].max() + 0.1*diff], color='red', linestyle='--')

    ax[0].set_xlabel(fr'$\rm Experimental \ {title} \ FE$')
    ax[0].set_ylabel(fr'$\rm Predicted \ {title} \ FE$')
    ax[0].legend(loc='best', ncol=2, fontsize=5)

    # bar plot with same coloring as scatter plot
    metric_value = [metric(data[f'{name}'], data[f'{pred_name}_1']),
        metric(data[f'{name}'], data[f'{pred_name}_2']),
        metric(data[f'{name}'], data[f'{pred_name}_3']),
        metric(data[f'{name}'], data[f'{pred_name}_4']),
        metric(data[f'{name}'], data[f'{pred_name}_5']),
        metric(data[f'{name}'], data[f'{pred_name}_6'])]
    
    # print(metric_value)

    ax[1].bar(['1', '2', '3', '4', '5', '6'], metric_value, color=['black', 'red', 'blue', 'green', 'purple', 'orange'])
    ax[1].set_ylabel(fr'$\rm {metric_name} \ (\%)$')
    ax[1].set_xlabel(r'$\rm Equation$')

    plt.tight_layout()

