"""
For each of the variables in 'main_cols', plot a boxplot of the whole data (X_all),
then a swarmplot of the 20 nearest neighbors' variable values (X_neigh),
and the values of the applicant customer (X_cust) as a pd.Series.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_boxplot_var_by_target(X_all, y_all, X_neigh, y_neigh, X_cust, main_cols, figsize=(15, 4)):

    df_all = pd.concat([X_all[main_cols], y_all.to_frame(name='TARGET')], axis=1)
    df_neigh = pd.concat([X_neigh[main_cols], y_neigh.to_frame(name='TARGET')], axis=1)
    df_cust = X_cust[main_cols].to_frame('values').reset_index()  # pd.Series to pd.DataFrame

    fig, ax = plt.subplots(figsize=figsize)

    # random sample of customers of the train set
    df_melt_all = df_all.reset_index()
    df_melt_all.columns = ['index'] + list(df_melt_all.columns)[1:]
    df_melt_all = df_melt_all.melt(id_vars=['index', 'TARGET'],  # SK_ID_CURR
                                   value_vars=main_cols,
                                   var_name="variables",
                                   value_name="values")
    sns.boxplot(data=df_melt_all, x='variables', y='values', hue='TARGET', linewidth=1,
                width=0.4, palette=['tab:green', 'tab:red'], showfliers=False, saturation=0.5,
                ax=ax)

    # 20 nearest neighbors
    df_melt_neigh = df_neigh.reset_index()
    df_melt_neigh.columns = ['index'] + list(df_melt_neigh.columns)[1:]
    df_melt_neigh = df_melt_neigh.melt(id_vars=['index', 'TARGET'],  # SK_ID_CURR
                                       value_vars=main_cols,
                                       var_name="variables",
                                       value_name="values")
    sns.swarmplot(data=df_melt_neigh, x='variables', y='values', hue='TARGET', linewidth=1,
                  palette=['darkgreen', 'darkred'], marker='o', edgecolor='k', ax=ax)

    # applicant customer
    df_melt_cust = df_cust.rename(columns={'index': "variables"})
    sns.swarmplot(data=df_melt_cust, x='variables', y='values', linewidth=1, color='y',
                  marker='o', size=10, edgecolor='k', label='applicant customer', ax=ax)

    # legend
    h, _ = ax.get_legend_handles_labels()
    ax.legend(handles=h[:5])

    plt.xticks(rotation=20, ha='right')
    plt.show()

    return fig


"""
Affiche les valeurs des clients en fonctions de deux paramètres en montrant leur classe
Compare l'ensemble des clients par rapport aux plus proches voisins et au client choisi.
X = données pour le calcul de la projection
ser_clust = données pour la classification des points (2 classes) (pd.Series)
n_display = items à tracer parmi toutes les données
plot_highlight = liste des index des plus proches voisins
X_cust = pd.Series des data de l'applicant customer
figsize=(10, 6) 
size=10
fontsize=12
columns=None : si None, alors projection sur toutes les variables, si plus de 2 projection
"""

from sklearn.manifold import trustworthiness
import matplotlib.pyplot as plt
import random
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def plot_scatter_projection(X, ser_clust, n_display, plot_highlight, X_cust, proj="PCA",
                            figsize=(10, 6), size=10, fontsize=12, columns=None):

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    X_all = pd.concat([X, X_cust.to_frame().T], axis=0)
    ind_neigh = list(plot_highlight.index)
    customer_idx = X_cust.name

    columns = X_all.columns if columns is None else columns

    if len(columns) == 2:
        # if only 2 columns passed
        df_data = X_all.loc[:, columns]
        ax.set_title('Two features compared', fontsize=fontsize + 2, fontweight='bold')
        ax.set_xlabel(columns[0], fontsize=fontsize)
        ax.set_ylabel(columns[1], fontsize=fontsize)

    elif len(columns) > 2:
        # if more than 2 columns passed, compute projection
        if proj=="t-SNE":
            estim = TSNE(n_components=2, random_state=14)
        else:
            estim = PCA(n_components=2, random_state=14)
            
        df_proj = pd.DataFrame(estim.fit_transform(X_all),
                               index=X_all.index,
                               columns=['proj' + str(i) for i in range(2)])
        
        trustw = trustworthiness(X_all, df_proj, n_neighbors=5, metric='euclidean')
        trustw = "{:.2f}".format(trustw)
        ax.set_title(f'{proj} projection (trustworthiness={trustw})',
                     fontsize=fontsize + 2, fontweight='bold')
        df_data = df_proj
        ax.set_xlabel("projection axis 1", fontsize=fontsize)
        ax.set_ylabel("projection axis 2", fontsize=fontsize)

    else:
        # si une colonne seulement
        df_data = pd.concat([X_all.loc[:, columns], X_all.loc[:, columns]], axis=1)
        ax.set_title('One feature', fontsize=fontsize + 2, fontweight='bold')
        ax.set_xlabel(columns[0], fontsize=fontsize)
        ax.set_ylabel(columns[0], fontsize=fontsize)

    # Showing points, cluster by cluster
    colors = ['green', 'red']
    for i, name_clust in enumerate(ser_clust.unique()):
        ind = ser_clust[ser_clust == name_clust].index

        if n_display is not None:
            display_samp = random.sample(set(list(X.index)), 200)
            ind = [i for i in ind if i in display_samp]
        # plot only a random selection of random sample points
        ax.scatter(df_data.loc[ind].iloc[:, 0],
                   df_data.loc[ind].iloc[:, 1],
                   s=size, alpha=0.7, c=colors[i], zorder=1,
                   label=f"Random sample ({name_clust})")
        # plot nearest neighbors
        ax.scatter(df_data.loc[ind_neigh].iloc[:, 0],
                   df_data.loc[ind_neigh].iloc[:, 1],
                   s=size * 5, alpha=0.7, c=colors[i], ec='k', zorder=3,
                   label=f"Nearest neighbors ({name_clust})")

    # plot the applicant customer
    ax.scatter(df_data.loc[customer_idx].iloc[0],
               df_data.loc[customer_idx].iloc[1],
               s=size * 10, alpha=0.7, c='yellow', ec='k', zorder=10,
               label="Applicant customer")

    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    ax.legend(prop={'size': fontsize - 2})

    return fig
