import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Plotting functions for training data.
# Saves Plots to data/Plots/

def draw_scatters(df, y, variables, n_rows, n_cols,
                  filename='plot'):
    """Plots each column of pandas dataframe against supervisor on
    a grid specified by n_rows x n_cols."""
    fig=plt.figure()
    for i, var_name in enumerate(variables):
        ax=fig.add_subplot(n_rows,n_cols,i+1)
        plt.scatter(df[var_name],y)
        ax.set_title(var_name)
    fig.tight_layout()  # Improves appearance a bit.
    plt.savefig(filename + '.png')


def draw_histograms(df, variables, n_rows, n_cols, bin_size=100,
                    filename='plot'):
    """Draws a grid of 100 bin histograms with n_rows x n_cols
    for each column of a pandas dataframe"""
    fig=plt.figure()
    for i, var_name in enumerate(variables):
        ax=fig.add_subplot(n_rows,n_cols,i+1)
        df[var_name].hist(bins=bin_size,ax=ax)
        ax.set_title(var_name+" Distribution")
    fig.tight_layout()  # Improves appearance a bit.
    plt.savefig(filename + '.png')


def sborn_heatmap(df,filename='plot',title='Plot'):
    """Plots and saves a seaborn heatmap to filename as a png"""
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,annot=True,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title(title, fontdict={'fontsize':27})
    plt.savefig(filename + '.png')


def missing_matrix(df, filename='plot'):
    """Creates missingness plot and saves it as png at filename"""
    msno.matrix(df)
    plt.savefig(filename + '.png')


def impute_scale_pca(df, components, filename='plot'):
    """Returns sklearn.PCA object for a dataset after imputing/scaling. Saves
    the plot of the first two PC scores at filename."""
    ImputeX_train = IterativeImputer(sample_posterior=True,
                                               imputation_order='descending',
                                               skip_complete=True,
                                               random_state=42).fit_transform(df)
    Scale_X = StandardScaler().fit_transform(ImputeX_train)
    pca = PCA(n_components=components)
    PC_scores = pca.fit_transform(Scale_X)
    plt.figure()
    plt.scatter(PC_scores[:,0],PC_scores[:,1])
    plt.title('PC1 vs PC2')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig(filename + '.png')
    return PC_scores


def main(datapath= '', supervisor = '', destpath = ''):

    if not os.path.exists(destpath):
        os.makedirs(destpath)

    df = pd.read_csv(datapath)

    x_train = df.drop([supervisor],axis=1)
    y_train  = df[supervisor]

    feat_names = x_train.columns.to_list()

    draw_scatters(x_train,y_train,feat_names,3,3,
                  filename=os.path.join(destpath, 'Scatter_Plots'))

    draw_histograms(x_train,feat_names,3,3,bin_size=50,
                    filename=os.path.join(destpath, 'Feature_Hists'))

    sborn_heatmap(x_train,filename=os.path.join(destpath, 'Corr_Heatmap'),
                  title='Correlation Heatmap (Training Data)')

    missing_matrix(x_train,filename=os.path.join(destpath, 'Missing_Data'))

    impute_scale_pca(x_train,components = 2,
                     filename=os.path.join(destpath, 'PC_Scores'))


if __name__ == '__main__':
    main(datapath='data/cleandata/Train.csv',
         supervisor='Power Source',
         destpath= 'data/Plots')
