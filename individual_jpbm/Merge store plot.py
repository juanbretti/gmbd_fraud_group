# %%
## Libraries ----
import pandas as pd
import glob
import os
import plotly.graph_objects as go
from sklearn import preprocessing
import pickle

# %%
## Merge the CSVs from grid seach ----
def merge_csv(source, target):
    try:
        os.remove(target)
    except:
        pass

    all_files = glob.glob(source)
    
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        df['Filename'] = filename
        li.append(df)
    frame = pd.concat(li, axis=0, ignore_index=True).sort_values(by='Gini', ascending=False)

    frame.to_csv(target, sep=',')

# Grid
merge_csv("../Store/Grid/*.csv", '../Store/submission_results_grid.csv')
# Original
merge_csv("../Store/24/*.csv", '../Store/submission_results_24.csv')

# %%
## Merge the two versions of the grid search  ----
df_grid = pd.read_csv('../Store/submission_results_grid.csv', index_col=None, header=0)
df_grid['Source'] = 'Grid'
df_grid['Model'] = 'RandomForestClassifier'

df_24 = pd.read_csv('../Store/submission_results_24.csv', index_col=None, header=0)
df_24['Source'] = '24'
df_24.rename(columns={"Strategy fill NA": "strategy_fill_na", 'Seed': 'seed'}, inplace=True)

df_24_description = pd.read_csv('../Store/Model description 24.csv', index_col=None, header=0)

df_24 = df_24.merge(df_24_description, on='Model')

df = pd.concat([df_grid, df_24], axis=0, ignore_index=True)

df = df.sort_values(by='Gini', ascending=False)

# %%
## Preprocessing ----
df['Bins'].fillna(0, inplace=True)
df['Bins icn'].fillna(0, inplace=True)
df['Bins ico'].fillna(0, inplace=True)
df['Bins if'].fillna(0, inplace=True)
df['bootstrap'].fillna(True, inplace=True)
df['class_weight'].fillna('None', inplace=True)
df['criterion'].fillna('gini', inplace=True)
df['Strategy'].fillna('Nothing', inplace=True)
df['strategy_fill_na'].fillna('0', inplace=True)

le = preprocessing.LabelEncoder()
df['strategy_fill_na_enc'] = le.fit_transform(df['strategy_fill_na'])
enc_strategy_fill_na_enc = list(le.classes_)
list_strategy_fill_na_enc = list(range(0, len(le.classes_)))

df['class_weight_enc'] = le.fit_transform(df['class_weight'])
enc_class_weight_enc = list(le.classes_)
list_class_weight_enc = list(range(0, len(le.classes_)))

df['criterion_enc'] = le.fit_transform(df['criterion'])
enc_criterion_enc = list(le.classes_)
list_criterion_enc = list(range(0, len(le.classes_)))

df['bootstrap_enc'] = le.fit_transform(df['bootstrap'])
enc_bootstrap_enc = list(le.classes_)
list_bootstrap_enc = list(range(0, len(le.classes_)))

df['Strategy_enc'] = le.fit_transform(df['Strategy'])
enc_Strategy_enc = list(le.classes_)
list_Strategy_enc = list(range(0, len(le.classes_)))

# n_bins = 20
# le = preprocessing.KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
# df['Gini_enc'] = le.fit_transform(df['Gini'].values.reshape(-1, 1))/n_bins

threshold = 0.534811
df['Gini_enc'] = [1 if x >= threshold else 0 for x in df['Gini']]

# %%
## Parallel coordinates plot from the grid search ----
# https://plotly.com/python/parallel-coordinates-plot/

fig = go.Figure(data=
    go.Parcoords(
        line = dict(color = df['Gini_enc'],
                    colorscale = 'bluered'),
        dimensions = list([
            dict(label = 'Bins', values = df['Bins']),
            dict(label = 'Bins icn', values = df['Bins icn']),
            dict(label = 'Bins ico', values = df['Bins ico']),
            dict(label = 'Bins if', values = df['Bins if']),

            dict(label = 'strategy_fill_na', values = df['strategy_fill_na_enc'], tickvals=list_strategy_fill_na_enc, ticktext = enc_strategy_fill_na_enc),
            dict(label = 'class_weight', values = df['class_weight_enc'], tickvals=list_class_weight_enc, ticktext = enc_class_weight_enc),
            dict(label = 'criterion', values = df['criterion_enc'], tickvals=list_criterion_enc, ticktext = enc_criterion_enc),
            dict(label = 'bootstrap', values = df['criterion_enc'], tickvals=list_bootstrap_enc, ticktext = enc_bootstrap_enc),
            dict(label = 'Strategy', values = df['Strategy_enc'], tickvals=list_Strategy_enc, ticktext = enc_Strategy_enc),
            dict(label = 'min_weight_fraction_leaf', values = df['min_weight_fraction_leaf']),
            dict(label = 'n_estimators', values = df['n_estimators']),


            dict(label = 'seed', values = df['seed']),
            dict(label = 'Gini', values = df['Gini']),
        ])
    )
)

fig.update_layout(
    autosize=False,
    width=1500,
    height=500,
    title=f'Gini higher than {threshold}')

fig.show()

# %%
## Performance per model algorithm ----
import seaborn as sns
import  matplotlib.pyplot as plt

def model_split(x):
    try:
        x = x.split(' ')[0]
    except:
        pass
    return x

df['Model simple'] = [model_split(x) for x in df['Model']]

df['Model simple'].value_counts()

a4_dims = (5, 3)
fig, ax = plt.subplots(figsize=a4_dims)
g = sns.boxplot(x="Model simple", y="Gini",
                data=df, ax=ax)

plt.xticks(rotation=90)

# %%