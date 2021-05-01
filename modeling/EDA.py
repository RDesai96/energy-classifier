import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import missingno as msno
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('data/cleandata/MergedComplete.csv')

df[['State', 'Power Source']] = df[['State', 'Power Source']].astype('category')
df_learn = df.drop(['Power Source','State'] ,axis=1)

X_train, X_test, y_train, y_test = \
    train_test_split(df_learn, df['Power Source'],
                     test_size=0.3,
                     random_state=42)

feat_names = X_train.columns.to_list()
X_train.reset_index(drop=True, inplace=True)

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


draw_scatters(X_train,y_train,feat_names,3,3,
              filename='data/Plots/Scatter_Plots')


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


draw_histograms(X_train,feat_names,3,3,bin_size=50,
                filename='data/Plots/Feature_Hists')


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


sborn_heatmap(X_train,filename='data/Plots/Corr_Heatmap',
              title='Correlation Heatmap (Training Data)')


def missing_matrix(df, filename='plot'):
    """Creates missingness plot and saves it as png at filename"""
    msno.matrix(df)
    plt.savefig(filename + '.png')

missing_matrix(X_train,filename='data/Plots/Missing_Data')


def impute_scale_pca(df, components, filename='plot'):
    """Returns sklearn.PCA object for a dataset after imputing/scaling. Saves
    the plot of the first two PC scores at filename."""
    ImputeX_train = KNNImputer(add_indicator=False).fit_transform(df)
    Scale_X = StandardScaler().fit_transform(ImputeX_train)
    pca = PCA(n_components=components)
    PC_scores = pca.fit_transform(Scale_X)
    plt.figure()
    plt.scatter(PC_scores[:,0],PC_scores[:,1])
    plt.title('PC1 vs PC2')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig(filename + '.png')
    return pca


impute_scale_pca(X_train,components = 2,
                 filename='data/Plots/PC_scores')

