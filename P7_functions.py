"""
Builds a customizable column_transformer which parameters can be optimized in a GridSearchCV
CATEGORICAL : three differents startegies for 3 different types of
categorical variables:
- low cardinality: customizable strategy (strat_low_card)
- high cardinality: customizable strategy (strat_high_card)
- boolean or equivalent (2 categories): ordinal
QUANTITATIVE (remainder):
- StandardScaler

-> EXAMPLE (to use apart from gscv):
cust_enc = CustTransformer(thresh_card=12,
                       strat_binary = 'ord',
                       strat_low_card = 'ohe',
                       strat_high_card = 'loo',
                       strat_quant = 'stand')
cust_enc.fit(X_tr, y1_tr)
cust_enc.transform(X_tr).shape, X_tr.shape

-> EXAMPLE (to fetch names of the modified dataframe):
small_df = df[['Outlier', 'Neighborhood', 'CertifiedPreviousYear',
               'NumberofFloors','ExtsurfVolRatio']]
# small_df.head(2)
cust_trans = CustTransformer()
cust_trans.fit(small_df)
df_enc = cust_trans.transform(small_df)
cust_trans.get_feature_names(small_df)
"""

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import category_encoders as ce
from sklearn.preprocessing import *
import numpy as np
import pandas as pd


class CustTransformer(BaseEstimator):

    def __init__(self, thresh_card=12,
                 strat_binary='ord', strat_low_card='ohe',
                 strat_high_card='bin', strat_quant='stand'):
        self.thresh_card = thresh_card
        self.strat_binary = strat_binary
        self.strat_low_card = strat_low_card
        self.strat_high_card = strat_high_card
        self.strat_quant = strat_quant
        self.dict_enc_strat = {'binary': strat_binary,
                               'low_card': strat_low_card,
                               'high_card': strat_high_card,
                               'numeric': strat_quant}
        self.cat_trans = None
        self.has_cat = None
        self.has_num = None
        self.ct_cat = None
        self.num_cols = None
        self.cat_cols = None
        self.name_columns = None
        self.column_trans = None
        self.num_trans = None

    def d_type_col(self, X):
        bin_cols = X.nunique()[X.nunique() <= 2].index
        X_C_cols = X.select_dtypes(include=['object', 'category'])
        C_l_card_cols = \
            X_C_cols.nunique()[X_C_cols.nunique()
                               .between(3, self.thresh_card)].index
        C_h_card_cols = \
            X_C_cols.nunique()[X_C_cols.nunique() > self.thresh_card].index
        Q_cols = [c for c in X.select_dtypes(include=[np.number]).columns
                  if c not in bin_cols]
        d_t = {'binary': bin_cols,
               'low_card': C_l_card_cols,
               'high_card': C_h_card_cols,
               'numeric': Q_cols}
        d_t = {k: v for k, v in d_t.items() if len(v)}
        # print(d_t)
        return d_t

    def get_feature_names(self, X, y=None):
        if self.has_num and self.has_cat:
            self.ct_cat.fit(X, y)
            cols = self.ct_cat.get_feature_names() + self.num_cols
        elif self.has_num and not self.has_cat:
            cols = self.num_cols
        elif not self.has_num and self.has_cat:
            self.ct_cat.fit(X, y)
            cols = self.ct_cat.get_feature_names()
        else:
            cols = None
        return cols

    def fit(self, X, y=None):
        # Dictionary to translate strategies
        d_enc = {'ohe': ce.OneHotEncoder(),
                 'hash': ce.HashingEncoder(),
                 'ord': ce.OrdinalEncoder(),
                 'loo': ce.LeaveOneOutEncoder(),
                 'bin': ce.BinaryEncoder(),
                 'stand': StandardScaler(),
                 'minmax': MinMaxScaler(),
                 'maxabs': MaxAbsScaler(),
                 'robust': RobustScaler(quantile_range=(25, 75)),
                 'norm': Normalizer(),
                 'quant_uni': QuantileTransformer(output_distribution='uniform'),
                 'quant_norm': QuantileTransformer(output_distribution='normal'),
                 'boxcox': PowerTransformer(method='box-cox'),
                 'yeo': PowerTransformer(method='yeo-johnson'),
                 'log': FunctionTransformer(func=lambda x: np.log1p(x),
                                            inverse_func=lambda x: np.expm1(x)),
                 'none': FunctionTransformer(func=lambda x: x,
                                             inverse_func=lambda x: x),
                 }

        # # dictionnaire liste des transfo categorielles EXISTANTES
        d_t = self.d_type_col(X)
        # numerics
        self.has_num = ('numeric' in d_t.keys())
        # categoricals
        self.has_cat = len([s for s in d_t.keys() if s in ['binary', 'low_card', 'high_card']]) > 0
        if self.has_cat:
            list_trans = []  # dictionnaire des transfo categorielles EXISTANTES
            for k, v in d_t.items():
                if k != 'numeric':
                    list_trans.append((k, d_enc[self.dict_enc_strat[k]], v))

            self.cat_cols = []  # liste des colonnes catégorielles à transformer
            for k, v in self.d_type_col(X).items():
                if k != 'numeric':
                    self.cat_cols += (list(v))

            self.ct_cat = ColumnTransformer(list_trans)
            self.cat_trans = Pipeline([("categ", self.ct_cat)])

        if self.has_num:
            self.num_trans = Pipeline([("numeric", d_enc[self.strat_quant])])
            self.num_cols = d_t['numeric']

        if self.has_num and self.has_cat:
            self.column_trans = \
                ColumnTransformer([('cat', self.cat_trans, self.cat_cols),
                                   ('num', self.num_trans, self.num_cols)])
        elif self.has_num and not self.has_cat:
            self.column_trans = \
                ColumnTransformer([('num', self.num_trans, self.num_cols)])
        elif not self.has_num and self.has_cat:
            self.column_trans = ColumnTransformer([('cat', self.cat_trans, self.cat_cols)])
        else:
            print("The dataframe is empty : no transformation can be done")
        self.name_columns = self.get_feature_names(X, y)
        return self.column_trans.fit(X, y)

    def transform(self, X, y=None):
        return pd.DataFrame(self.column_trans.transform(X),
                            index=X.index,
                            columns=self.name_columns)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return pd.DataFrame(self.column_trans.transform(X),
                            index=X.index,
                            columns=self.name_columns)


"""
For each of the variables in 'main_cols', plot a boxplot of the whole data (X_all),
then a swarmplot of the 20 nearest neighbors' variable values (X_neigh),
and the values of the applicant customer (X_cust) as a pd.Series.
"""

import seaborn as sns


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

    plt.xticks(rotation=20)
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


def plot_scatter_projection(X, ser_clust, n_display, plot_highlight, X_cust,
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
        # if more than 2 columns passed, compute T-SNE projection
        tsne = TSNE(n_components=2, random_state=14)
        df_proj = pd.DataFrame(tsne.fit_transform(X_all),
                               index=X_all.index,
                               columns=['t-SNE' + str(i) for i in range(2)])
        trustw = trustworthiness(X_all, df_proj, n_neighbors=5, metric='euclidean')
        trustw = "{:.2f}".format(trustw)
        ax.set_title(f't-SNE projection (trustworthiness={trustw})',
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
