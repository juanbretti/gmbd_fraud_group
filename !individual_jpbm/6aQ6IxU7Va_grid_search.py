"""
Welcome to a competition powered by AutoDSC for Data Science Challenges! By Prof. Manoel Gadi!

Simply run this code and start competing today in the competion: 6aQ6IxU7Va

6aQ6IxU7Va details:
 - Description / Descripción: FRAUD MODELLING CHALLENGE - Predict which Credit Card Application is legitimate and which belongs to a fraudster instead.
 - Maximum number of daily attempts / Número máximo de intentos diarios: 10000
 - Creation date / Fecha de creación: 2020-06-10 11:36:52
 - Starting date / Fecha de inicio: 2021-05-12 12:00:00
 - Ending date / Fecha de fin: 2021-05-30 23:59:00
 - Minimum time between prediction submissions / Tiempo mínimo entre envíos de predicciones: 30

Of course, to win the competition you should improve the starting model! So let's get to work!
"""
# %% 
## Setup ----
### Libraries ----
# General usage
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from profmanoelgadi_support_package import IV, PSI
import profmanoelgadi_support_package as manoel
import itertools
import random
import pickle
import warnings
import time

# Model management
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from sklearn.metrics import make_scorer

# Available models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import class_weight
import xgboost as xgb
import lightgbm as lgb
from sklearn.utils import class_weight

# Reporting
from pandas_profiling import ProfileReport
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, classification_report

# Preprocessing
from sklearn.preprocessing import RobustScaler, KBinsDiscretizer, StandardScaler
from sklearn.impute import SimpleImputer

# %%
### Constants ----
# Random state
RANDOM_STATE = 42
# Filter
FILTER_EXCLUDE = ['Unnamed: 0', 'id', 'contract_date', 'ib_var_12']
FILTER_TARGET = 'ob_target'
# Time between submissions
TIME_WAIT_SUBMIT = 61
TIME_WAIT_RETRY = 10

### General config ----
pd.set_option('display.max_colum', 100)
pd.set_option('display.max_row', 100)
warnings.filterwarnings("ignore", message="Bins whose width are too small")

# %% 
## Loading data ----
try: # reading train csv from local file
    df_train = pd.read_csv("../Data/mfalonso__6aQ6IxU7Va__train.csv")
    df_train.head()
except: # reading train csv from the internet if it is the first time
    import urllib
    csv_train = urllib.request.urlopen("http://manoelutad.pythonanywhere.com/static/uploads/mfalonso__6aQ6IxU7Va__train.csv")
    csv_train_content = csv_train.read()
    with open("../Data/mfalonso__6aQ6IxU7Va__train.csv", 'wb') as f:
            f.write(csv_train_content)
    df_train = pd.read_csv("../Data/mfalonso__6aQ6IxU7Va__train.csv")
    df_train.head()

try: # reading test csv from local file
    df_test = pd.read_csv("../Data/mfalonso__6aQ6IxU7Va__test.csv")
    df_test.head()
except: # reading test csv from the internet if it is the first time
    import urllib
    csv_test = urllib.request.urlopen("http://manoelutad.pythonanywhere.com/static/uploads/mfalonso__6aQ6IxU7Va__test.csv")
    csv_test_content = csv_test.read()
    with open("../Data/mfalonso__6aQ6IxU7Va__test.csv", 'wb') as f:
            f.write(csv_test_content)
    df_test = pd.read_csv("../Data/mfalonso__6aQ6IxU7Va__test.csv")
    df_test.head()

# %%
## Columns ----
def df_columns(x):
    return df_train.filter(regex=x).columns.tolist()

# List of columns per class
filter_ib = df_columns('ib_var')
filter_icn = df_columns('icn_var')
filter_ico = df_columns('ico_var')
filter_if = df_columns('if_var')
# filter_include = filter_include_ga  # This is defined at the bottom of the code
filter_include = df_train.columns.tolist()

def filter_intersect_exclude(filter_ib, filter_include=filter_include, filter_exclude=FILTER_EXCLUDE):
    base = pd.Series(filter_ib)
    intersect = filter_include
    exclude = filter_exclude
    intersected = base[base.isin(intersect)]
    excluded = intersected[~intersected.isin(exclude)]
    return excluded

## Fill NA ----
# Impute mean to missing values in test
def fill_na(df_test, strategy='most_frequent'):
    if strategy == 'most_frequent':
        test_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    else:
        test_imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=strategy)
    df_filled = df_test.drop(FILTER_EXCLUDE, axis=1, errors='ignore')
    df_filled = test_imputer.fit_transform(df_filled)
    df_filled = pd.DataFrame(df_filled, columns=df_test.drop(FILTER_EXCLUDE, axis=1, errors='ignore').columns)
    return df_filled

## Transformations ----
def transform_scaler(df, encoder=None):
    if encoder is None:
        encoder = StandardScaler().fit(df)
    df_tranformed = encoder.transform(df)
    df_tranformed = pd.DataFrame(df_tranformed, columns=df.columns)
    return df_tranformed, encoder

def transform_binning(df, encoder=None, n_bins=None):
    if encoder is None:
        encoder = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        encoder = encoder.fit(df)
    df_tranformed = encoder.transform(df)
    df_tranformed = pd.DataFrame(df_tranformed, columns=df.columns)
    return df_tranformed, encoder

def transform_dummy(df):
    df_tranformed = pd.get_dummies(df, columns = df.columns, drop_first=True)
    return df_tranformed

def transform_apply(df, columns, transform_scaler_do=False, encoder_scaler=None, transform_binning_do=False, encoder_binning=None, n_bins=None, transform_dummy_do=False):

    filter_ = filter_intersect_exclude(columns)
    df_tranformed = df[filter_]
    
    if transform_scaler_do:
        if encoder_scaler is None:
            df_tranformed, encoder_scaler = transform_scaler(df_tranformed)
        else:
            df_tranformed = transform_scaler(df_tranformed, encoder=encoder_scaler)[0]

    if transform_binning_do:
        if encoder_binning is None:
            df_tranformed, encoder_binning = transform_binning(df_tranformed, n_bins=n_bins)
        else:
            df_tranformed = transform_binning(df_tranformed, encoder=encoder_binning)[0]
    
    if transform_dummy_do:
        df_tranformed = transform_dummy(df_tranformed)

    return df_tranformed, encoder_scaler, encoder_binning

def transform_apply_strategy(df_train, df_test, columns, strategy, n_bins=None):
    if strategy=='Nothing':
        df_train_transformed, encoder_scaler, encoder_binning = transform_apply(df_train, columns, transform_scaler_do=False, transform_binning_do=False, transform_dummy_do=False)
        df_test_transformed = transform_apply(df_test, columns, transform_scaler_do=False, transform_binning_do=False, transform_dummy_do=False)[0]
    elif strategy=='StandardScaler':
        df_train_transformed, encoder_scaler, encoder_binning = transform_apply(df_train, columns, transform_scaler_do=True, transform_binning_do=False, transform_dummy_do=False)
        df_test_transformed = transform_apply(df_test, columns, transform_scaler_do=True, encoder_scaler=encoder_scaler)[0] 
    elif strategy=='StandardScaler+KBinsDiscretizer':
        df_train_transformed, encoder_scaler, encoder_binning = transform_apply(df_train, columns, transform_scaler_do=True, transform_binning_do=True, transform_dummy_do=False, n_bins=n_bins)
        df_test_transformed = transform_apply(df_test, columns, transform_scaler_do=True, encoder_scaler=encoder_scaler, transform_binning_do=True, encoder_binning=encoder_binning)[0] 
    elif strategy=='StandardScaler+KBinsDiscretizer+Dummy':
        df_train_transformed, encoder_scaler, encoder_binning = transform_apply(df_train, columns, transform_scaler_do=True, transform_binning_do=True, transform_dummy_do=True, n_bins=n_bins)
        df_test_transformed = transform_apply(df_test, columns, transform_scaler_do=True, encoder_scaler=encoder_scaler, transform_binning_do=True, encoder_binning=encoder_binning, transform_dummy_do=True)[0] 
    elif strategy=='KBinsDiscretizer+Dummy':
        df_train_transformed, encoder_scaler, encoder_binning = transform_apply(df_train, columns, transform_scaler_do=False, transform_binning_do=True, transform_dummy_do=True, n_bins=n_bins)
        df_test_transformed = transform_apply(df_test, columns, transform_scaler_do=False, encoder_scaler=encoder_scaler, transform_binning_do=True, encoder_binning=encoder_binning, transform_dummy_do=True)[0] 
    else:
        pass
    return df_train_transformed, df_test_transformed

def df_transform_bins(df_train, df_test, strategy, bins_icn, bins_ico, bins_if, strategy_fill_na='most_frequent'):
    print('\n'*2)
    print(f'** Strategy `{strategy}`. Fill NA `{strategy_fill_na}`. Bins icn `{bins_icn}`, ico `{bins_ico}`, if `{bins_if}`')

    df_train_transformed_ib, df_test_transformed_ib = transform_apply_strategy(df_train, fill_na(df_test, strategy_fill_na), columns=filter_ib, strategy='Nothing')
    df_train_transformed_icn, df_test_transformed_icn = transform_apply_strategy(df_train, fill_na(df_test, strategy_fill_na), columns=filter_icn, strategy=strategy, n_bins=bins_icn)
    df_train_transformed_ico, df_test_transformed_ico = transform_apply_strategy(df_train, fill_na(df_test, strategy_fill_na), columns=filter_ico, strategy=strategy, n_bins=bins_ico)
    df_train_transformed_if, df_test_transformed_if = transform_apply_strategy(df_train, fill_na(df_test, strategy_fill_na), columns=filter_if, strategy=strategy, n_bins=bins_if)

    df_train_transformed = pd.concat([df_train_transformed_ib, df_train_transformed_icn, df_train_transformed_ico, df_train_transformed_if], axis=1)
    df_test_transformed = pd.concat([df_test_transformed_ib, df_test_transformed_icn, df_test_transformed_ico, df_test_transformed_if], axis=1)

    # Data split
    X_train = df_train_transformed
    y_train = df_train[FILTER_TARGET]
    X_test = df_test_transformed
    # Fix missing columns
    X_train[X_test.columns.drop(X_train.columns)] = 0

    return X_train, y_train, X_test

## Auxiliary functions ----
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

## Submission ----
def prediction_score(y_train, pred_train, report=False):
    print ("STEP 4: ASSESSING THE MODEL...")
    gini_score = 2*roc_auc_score(y_train, pred_train)-1
    print ("GINI DEVELOPMENT=", gini_score)
    if report:
        print(classification_report(y_train, (pred_train>=0.5)*1))

def prediction_submission(id, pred_test):
    print ("STEP 5: SUBMITTING THE RESULTS...  DO NOT CHANGE THIS PART!")
    import requests
    from requests.auth import HTTPBasicAuth
    df_test_tosend = pd.DataFrame({'id': id, 'pred': pred_test})

    filename = "df_test_tosend.csv"
    df_test_tosend.to_csv(filename, sep=',')
    url = 'http://manoelutad.pythonanywhere.com/uploadpredictions/6aQ6IxU7Va'
    files_ = {'file': (filename, open(filename, 'rb')),
            'pycode': (__file__, open(__file__, 'rb'))}

    submission_gini_score = None
    while submission_gini_score is None:
        try:
            rsub = requests.post(url, files=files_, auth=HTTPBasicAuth("juanbretti", "sha256$F7s0Yak4$c326c18e2afb65348d7c462ba09973e414380c36bab47c8439d3f193c76a3f94"))
            resp_str = str(rsub.text)
            print ("RESULT SUBMISSION:", resp_str)  
            start_time = datetime.now()
            submission_gini_score = float(resp_str.split(' = ')[1])
        except:
            files_ = {'file': (filename, open(filename, 'rb')),
                      'pycode': (__file__, open(__file__, 'rb'))}
            time.sleep(TIME_WAIT_RETRY)

    return submission_gini_score, start_time

# %%
## Testing models ----
def models_list(seed=RANDOM_STATE):
    print('\n'*1)
    print(f'** Testing seed `{seed}`')
    # Available models
    models = [
        # Best models
        # ('RandomForestClassifier', RandomForestClassifier(random_state=seed, n_jobs=-1, class_weight='balanced_subsample')), 
        # ('RandomForestClassifier n_estimators=100', RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1, class_weight='balanced_subsample')), 
        # ('RandomForestClassifier n_estimators=300 StandardScaler=off', RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1, class_weight='balanced_subsample')), 
        # ('RandomForestClassifier n_estimators=300 bootstrap=False StandardScaler=off', RandomForestClassifier(n_estimators=300, bootstrap=False, random_state=seed, n_jobs=-1, class_weight='balanced_subsample')), 
        # ('RandomForestClassifier n_estimators=1000 StandardScaler=off', RandomForestClassifier(n_estimators=1000, random_state=seed, n_jobs=-1, class_weight='balanced_subsample')), 
        # ('RandomForestClassifier n_estimators=1000 bootstrap=False', RandomForestClassifier(n_estimators=1000, bootstrap=False, random_state=seed, n_jobs=-1, class_weight='balanced_subsample')), 
        # ('RandomForestClassifier n_estimators=1000 bootstrap=False min_weight_fraction_leaf=0.01', RandomForestClassifier(n_estimators=1000, min_weight_fraction_leaf=0.01, bootstrap=False, random_state=seed, n_jobs=-1, class_weight='balanced_subsample')), 
        # ('RandomForestClassifier n_estimators=1000 bootstrap=False', RandomForestClassifier(n_estimators=1000, bootstrap=False, random_state=seed, n_jobs=-1, class_weight='balanced_subsample')), 
        # ('RandomForestClassifier n_estimators=1000 min_weight_fraction_leaf=0.0001', RandomForestClassifier(n_estimators=1000, min_weight_fraction_leaf=0.0001, random_state=seed, n_jobs=-1, class_weight='balanced_subsample')), 
        # ('RandomForestClassifier n_estimators=1000 bootstrap=False min_weight_fraction_leaf=0.0001 criterion=entropy', RandomForestClassifier(n_estimators=1000, criterion='entropy', min_weight_fraction_leaf=0.0001, bootstrap=False, random_state=seed, n_jobs=-1, class_weight='balanced_subsample')), 
        # ('RandomForestClassifier n_estimators=1000 bootstrap=False min_weight_fraction_leaf=0.0001', RandomForestClassifier(n_estimators=1000, min_weight_fraction_leaf=0.0001, bootstrap=False, random_state=seed, n_jobs=-1, class_weight='balanced_subsample')), 
        ('RandomForestClassifier n_estimators=1000 bootstrap=False min_weight_fraction_leaf=0.0001', RandomForestClassifier(n_estimators=1000, min_weight_fraction_leaf=0.0001, bootstrap=False, random_state=seed, n_jobs=-1, class_weight='balanced_subsample')), 
        # ('RandomForestClassifier n_estimators=1000 bootstrap=False', RandomForestClassifier(n_estimators=1000, bootstrap=False, random_state=seed, n_jobs=-1, class_weight='balanced_subsample')), 
        # ('RandomForestClassifier n_estimators=10000 bootstrap=False', RandomForestClassifier(n_estimators=10000, bootstrap=False, random_state=seed, n_jobs=-1, class_weight='balanced_subsample')), 
        # ('RandomForestClassifier n_estimators=1000', RandomForestClassifier(n_estimators=1000, random_state=seed, n_jobs=-1, class_weight='balanced_subsample')), 
        # ('RandomForestClassifier n_estimators=2000', RandomForestClassifier(n_estimators=2000, random_state=seed, n_jobs=-1, class_weight='balanced_subsample')), 
        # ('RandomForestClassifier n_estimators=1000 min_weight_fraction_leaf=0.0001 criterion=entropy', RandomForestClassifier(n_estimators=1000, criterion='entropy', min_weight_fraction_leaf=0.0001, random_state=seed, n_jobs=-1, class_weight='balanced_subsample')), 
        # ('RandomForestClassifier n_estimators=1000 bootstrap=False StandardScaler=off', RandomForestClassifier(n_estimators=1000, bootstrap=False, random_state=seed, n_jobs=-1, class_weight='balanced_subsample')), 
        # ('RandomForestClassifier n_estimators=2000 StandardScaler=off', RandomForestClassifier(n_estimators=2000, random_state=seed, n_jobs=-1, class_weight='balanced_subsample')), 
        # ('RandomForestClassifier n_estimators=2000 bootstrap=False StandardScaler=off', RandomForestClassifier(n_estimators=2000, bootstrap=False, random_state=seed, n_jobs=-1, class_weight='balanced_subsample')), 
        # ('RandomForestClassifier n_estimators=2000 max_features=None', RandomForestClassifier(n_estimators=2000, max_features=None, random_state=seed, n_jobs=-1, class_weight='balanced_subsample')), 
        # ('RandomForestClassifier n_estimators=1000 bootstrap=False max_features=None', RandomForestClassifier(n_estimators=1000, bootstrap=False, max_features=None, random_state=seed, n_jobs=-1, class_weight='balanced_subsample')), 
        # ('RandomForestClassifier n_estimators=1000 max_depth=10', RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=seed, n_jobs=-1, class_weight='balanced_subsample')), 
        # ('ExtraTreesClassifier n_estimators=2000', ExtraTreesClassifier(n_estimators=2000, random_state=seed, n_jobs=-1, class_weight='balanced_subsample')), 
        # ('ExtraTreesClassifier n_estimators=2000 bootstrap=False', ExtraTreesClassifier(n_estimators=2000, bootstrap=False, random_state=seed, n_jobs=-1, class_weight='balanced_subsample')), 
        # ('SVC', SVC(random_state=seed, class_weight='balanced', probability=True)), 
        # ('LGBMClassifier', lgb.LGBMClassifier()), 
        # Not good
        # ('AdaBoostClassifier', AdaBoostClassifier(n_estimators=2000, random_state=seed)), 
        # ('GradientBoostingClassifier', GradientBoostingClassifier(n_estimators=20, random_state=seed)), 
        # ('KNeighborsClassifier', KNeighborsClassifier(n_neighbors=10, n_jobs=-1)), 
        # ('LogisticRegression', LogisticRegression(random_state=seed, n_jobs=-1, class_weight='balanced_subsample')), 
        # ('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()), 
        # Fails
        # ('XGBClassifier', xgb.XGBClassifier(n_estimators=10, objective ='binary:logistic', tree_method='gpu_hist', gpu_id=0, nthread=-1, random_state=seed)),
        # ('GaussianNB', GaussianNB()),
        # ('QuadraticDiscriminantAnalysis', QuadraticDiscriminantAnalysis()), 
    ]
    return models

def test_name_model(start_time, name, model, X_train, y_train, X_test):
    print('\n'*1)
    print(f'** Testing model `{name}`')
    model.fit(X_train, y_train)
    pred_train = model.predict_proba(X_train)[:,1]
    pred_test  = model.predict_proba(X_test)[:,1]
    prediction_score(y_train, pred_train)
            
    # Wait 60 seconds between submissions
    while (datetime.now()-start_time).seconds < TIME_WAIT_SUBMIT:
        time.sleep(1)

    # Submit
    submission_gini_score, start_time = prediction_submission(df_test['Unnamed: 0'], pred_test)

    return submission_gini_score, start_time

# %%
## Iteration ----
# Prepare the containers
submission_results = pd.DataFrame()
start_time = datetime.now()-timedelta(seconds=TIME_WAIT_SUBMIT)

# Ranges for the bins
range_strategy = ['KBinsDiscretizer+Dummy', 'StandardScaler+KBinsDiscretizer+Dummy']
range_strategy_fill_na = [0, 'most_frequent']
range_bins_icn = [3, 4]
range_bins_ico = [4, 8]
range_bins_if = [15, 17, 20]
range_seed = range(100, 200)
range_total1 = list(itertools.product(range_strategy, range_strategy_fill_na, range_bins_icn, range_bins_ico, range_bins_if, range_seed))

range_strategy = ['Nothing']
range_strategy_fill_na = [0, 'most_frequent']
range_bins_icn = [None]
range_bins_ico = [None]
range_bins_if = [None]
range_seed = range(300, 400)
range_total2 = list(itertools.product(range_strategy, range_strategy_fill_na, range_bins_icn, range_bins_ico, range_bins_if, range_seed))

range_total = range_total1 + range_total2

# Random list
random.shuffle(range_total, random.random)

# Start the iteration
for strategy, strategy_fill_na, bins_icn, bins_ico, bins_if, seed in range_total:
    X_train, y_train, X_test = df_transform_bins(df_train, df_test, strategy, bins_icn, bins_ico, bins_if, strategy_fill_na)
    
    models = models_list(seed)

    for name, model in models:
        # Testing the model
        submission_gini, start_time = test_name_model(start_time, name, model, X_train, y_train, X_test)

        # Store results
        submission_results = submission_results.append(pd.DataFrame({
            'Model': [name],
            'Gini': [submission_gini],
            'Submission time': [start_time],
            'Seed': [seed],
            'Strategy': [strategy],
            'Strategy fill NA': [strategy_fill_na],
            'Bins icn': [bins_icn],
            'Bins ico': [bins_ico],
            'Bins if': [bins_if]}), ignore_index=True)

        ### Emergency store ----
        submission_results.to_pickle('../Store/submission_results-temp.pkl')

# %%
### Emergency restore ----
submission_results = pd.read_pickle('../Store/submission_results-temp.pkl')

### Results ----
# Present the results
submission_results = submission_results.sort_values(by='Gini', ascending=False)
# Store the results
timestr = time.strftime("%Y%m%d-%H%M%S")
submission_results.to_csv(f'../Store/submission_results-{timestr}.csv', sep=',')
# View the results
submission_results

# %%
# http://manoelutad.pythonanywhere.com/ranking/6aQ6IxU7Va

# %%
## Run the best model ----
name = 'Best model'
strategy = 'Nothing'
bins_icn, bins_ico, bins_if = None, None, None
strategy_fill_na = 'most_frequent'
seed = 252

X_train, y_train, X_test = df_transform_bins(df_train, df_test, strategy, bins_icn, bins_ico, bins_if, strategy_fill_na)
model = RandomForestClassifier(n_estimators=1000, min_weight_fraction_leaf=0.0001, bootstrap=False, random_state=seed, n_jobs=-1, class_weight='balanced_subsample')

start_time = datetime.now()-timedelta(seconds=30)
submission_gini, start_time = test_name_model(start_time, name, model, X_train, y_train, X_test)

pickle.dump(model, open('../Store/model-temp.pkl', 'wb'))
model = pickle.load(open('../Store/model-temp.pkl', 'rb'))

# RESULT SUBMISSION: Competition / competición: 6aQ6IxU7Va - gini = 0.5423629146295255

# %%
