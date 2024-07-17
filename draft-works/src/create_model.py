import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import yaml
import shap

from utils import load_data
from model import construct_and_fit_gp_model
from plotter import Visualization #disp_shap_bar, show_crossplot


viz = Visualization()

def main(args):
    
    # load the data
    data, features_col, target_col = load_data(args.data_path)

    # set the seed
    if args.seed is not None:
        seed = args.seed
    else:
        seed = np.random.randint(0, 2**32 - 1)
    print('Seed: ', seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # shuffle the data
    if args.reshuffle:
        index = np.random.permutation(data.index)
        print('Data reshuffled')
        data = data.loc[index].reset_index(drop=True)
        

    # fit model
    X = data[features_col].values
    y = data[target_col[args.label_index]].values
    print(f'Target variable: {target_col[args.label_index]} selected')

    # total number of samples in data set
    nb_samples = X.shape[0]
    # split
    ids_train, ids_test = train_test_split(range(nb_samples), test_size=args.test_size, random_state=seed)

    # train surrogate model for test data, on acquired set up till top COF was found.
    X_train = torch.from_numpy(X[ids_train, :])
    X_test = torch.from_numpy(X[ids_test, :])

    y_train = torch.from_numpy(y[ids_train].reshape(-1, 1))
    y_test = torch.from_numpy(y[ids_test].reshape(-1, 1))

    print(f'Shape of X_train: {X_train.shape}, y_train: {y_train.shape}')
    print(f'Shape of X_test: {X_test.shape}, y_test: {y_test.shape}')

    gp = construct_and_fit_gp_model(X_train, y_train)
    y_train_pred = gp.posterior(X_train)
    y_test_pred = gp.posterior(X_test)

    y_train_pred_mean = y_train_pred.mean.squeeze().detach().numpy()
    y_test_pred_mean = y_test_pred.mean.squeeze().detach().numpy()

    y_train_pred_sttd = y_train_pred.stddev.squeeze().detach().numpy()
    y_test_pred_sttd = y_test_pred.stddev.squeeze().detach().numpy()

    results = {
                'train': {'y': y_train, 'y_pred': y_train_pred_mean, 'y_pred_sttd': y_train_pred_sttd},
                'test': {'y': y_test, 'y_pred': y_test_pred_mean, 'y_pred_sttd': y_test_pred_sttd},
                }

    # add R2 and MAE to plot
    r2_train = r2_score(y_train, y_train_pred_mean)
    r2_test = r2_score(y_test, y_test_pred_mean)
    abserr_train = mean_absolute_error(y_train, y_train_pred_mean)
    abserr_test = mean_absolute_error(y_test, y_test_pred_mean)
    rmse_train = mean_squared_error(y_train, y_train_pred_mean, squared=False)
    rmse_test = mean_squared_error(y_test, y_test_pred_mean, squared=False)

    # print the results
    print(f'R2:- Train = {r2_train:.3f} Test = {r2_test:.3f}')
    print(f'MAE:- Train = {abserr_train:.3f} Test = {abserr_test:.3f}')
    print(f'RMSE:- Train = {rmse_train:.3f} Test = {rmse_test:.3f}')

    viz.show_crossplot(result=results, save=args.save, add_bar=args.add_bar, path=os.path.join(args.figures_path, 'crossplot', f'gp_model_id{args.label_index}_seed{args.seed}_v2.png'), label=args.label)

    # show the importance of the features
    def gp_predict(X):
        gp.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float)
            preds = gp(X)
            return preds.mean.numpy()
    
    explainer = shap.Explainer(gp_predict, X)
    shap_values = explainer(X)
    # #print(shap_values)
    # shap.plots.heatmap(shap_values)
    # # get yticks and set to features_col
    # plt.get_yticks()
    # plt.yticks(ticks=range(len(features_col)), labels=features_col)
    # viz.disp_shap_bar(shap_values=shap_values, data=X, names=features_col, color = 'blue', save=args.save, tag=args.label,
    #               path=os.path.join(args.figures_path, 'shap', f'shap_gp_model_id{args.label_index}_seed{args.seed}_v2.png'))
    
    # viz.disp_shap_bee(shap_values=shap_values, data=X, names=features_col, color = 'blue', save=args.save, tag=args.label,
    #               path=os.path.join(args.figures_path, 'shap', f'shap_bee_gp_model_id{args.label_index}_seed{args.seed}_v2.png'))
    
    

    # save the model
    if args.save:
        # save model
        torch.save(gp.state_dict(), os.path.join(args.models_path, f'gp_model_id{args.label_index}_seed{args.seed}_v2.pt'))

    # write args to a yaml file
    if args.save:
        with open(os.path.join(args.models_path, f'args_id{args.label_index}_seed{seed}_v2.yaml'), 'w') as file:
            yaml.dump(vars(args), file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate a Gaussian Process model.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data file')
    parser.add_argument('--label', type=str, default='C_2H_4', help='Label for the target variable')
    parser.add_argument('--label_index', type=int, default=0, help='Label index for the target variable')
    parser.add_argument('--seed', type=int, default=None, help='Seed for the random number generator')
    parser.add_argument('--figures_path', type=str, default='./reports', help='Path to save the plots')
    parser.add_argument('--models_path', type=str, default='./models', help='Path to save the model')
    parser.add_argument('--test_size', type=float, default=0.1, help='Size of the test set')
    parser.add_argument('--save', action='store_true', help='Save the results')
    parser.add_argument('--reshuffle', action='store_true', help='Reshuffle the data')
    parser.add_argument('--add_bar', action='store_true', help='Add a bar to the plot')
    args = parser.parse_args()

    main(args)
