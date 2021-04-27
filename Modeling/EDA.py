import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno


df = pd.read_csv('data/cleandata/MergedComplete.csv')
df_labels = df.columns.to_list()

def draw_histograms(df, variables, n_rows, n_cols, bin_size=100):
    """Draws a grid of 100 bin histograms with n_rows x n_cols"""
    fig=plt.figure()
    for i, var_name in enumerate(variables):
        ax=fig.add_subplot(n_rows,n_cols,i+1)
        df[var_name].hist(bins=bin_size,ax=ax)
        ax.set_title(var_name+" Distribution")
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()

draw_histograms(df,['Power Source','Longitude', 'Latitude',
                    'Capacity','GHI','DNI','a30','a35','a40'],3,3,bin_size=50)
plt.title('Feature Histograms')
plt.savefig('data/Plots/histograms.png')

def sborn_heatmap(df,filename='plot',title='Plot'):
    """Plots and saves a seaborn heatmap to filename as a png"""
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title(title)
    plt.savefig(filename + '.png')
sborn_heatmap(df,filename='data/Plots/correlation',
              title='Correlation Plot')

def missing_matrix(df, filename='plot', title='Plot'):
    """Creates missingness plot and saves it as png at filename"""
    msno.matrix(df)
    plt.title(title)
    plt.savefig(filename + '.png')
missing_matrix(df,filename='data/Plots/missingdata',
               title='Missing Values')

# TODO Screen for outliers
