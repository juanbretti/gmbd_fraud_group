# %%
## Constants ----
FILTER_EXCLUDE = ['Unnamed: 0', 'id', 'contract_date', 'ib_var_12']
FILTER_TARGET = 'ob_target'
TIME_WAIT_SUBMIT = 25
TIME_WAIT_RETRY = 1
RANDOM_SEED = 42

# Path
import os
script_dir = os.path.dirname(__file__)

print ("IMPORTING LIBRARIES...")
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
import time
from datetime import datetime, timedelta
import requests
from requests.auth import HTTPBasicAuth
from sklearn.impute import SimpleImputer
import itertools
import random

print ("LOADING DATASETS...")
try: # reading train csv from local file
    df_train = pd.read_csv(os.path.join(script_dir,"../Data/mfalonso__6aQ6IxU7Va__train.csv"))
    df_train.head()
except: # reading train csv from the internet if it is the first time
    import urllib
    csv_train = urllib.request.urlopen("http://manoelutad.pythonanywhere.com/static/uploads/mfalonso__6aQ6IxU7Va__train.csv")
    csv_train_content = csv_train.read()
    with open(os.path.join(script_dir,"../Data/mfalonso__6aQ6IxU7Va__train.csv"), 'wb') as f:
            f.write(csv_train_content)
    df_train = pd.read_csv(os.path.join(script_dir,"../Data/mfalonso__6aQ6IxU7Va__train.csv"))
    df_train.head()

try: # reading test csv from local file
    df_test = pd.read_csv(os.path.join(script_dir,"../Data/mfalonso__6aQ6IxU7Va__test.csv"))
    df_test.head()
except: # reading test csv from the internet if it is the first time
    import urllib
    csv_test = urllib.request.urlopen("http://manoelutad.pythonanywhere.com/static/uploads/mfalonso__6aQ6IxU7Va__test.csv")
    csv_test_content = csv_test.read()
    with open(os.path.join(script_dir,".../Data/mfalonso__6aQ6IxU7Va__test.csv"), 'wb') as f:
            f.write(csv_test_content)
    df_test = pd.read_csv(os.path.join(script_dir,"../Data/mfalonso__6aQ6IxU7Va__test.csv"))
    df_test.head()

print ("STEP 1: DOING MY TRANSFORMATIONS...")
def fill_na(df_test, strategy='most_frequent'):
    if strategy == 'most_frequent':
        test_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    else:
        test_imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=strategy)
    df_filled = test_imputer.fit_transform(df_test)
    df_filled = pd.DataFrame(df_filled, columns=df_test.columns)
    return df_filled

print ("STEP 2: SELECTING CHARACTERISTICS TO ENTER INTO THE MODEL...")

# Columns from the Genetic Algorithm
# 2000 = 0.5802179317738612
# 1500 = 0.5795444042920914
# 1000 = 0.5802618323644393
# 900 = 0.5815337942125158
# 850 = 0.5821668869398053
# 800 = 0.5824799674673518 # Sweet spot
# 750 = 0.5816793593286445
# 700 = 0.5818792225436464
in_model = ['ib_var_2',
'ib_var_3',
'ib_var_19',
'ib_var_21',
'icn_var_23',
'icn_var_24',
'ico_var_33',
'ico_var_34',
'ico_var_35',
'ico_var_37',
'ico_var_38',
'ico_var_41',
'ico_var_43',
'ico_var_45',
'ico_var_48',
'ico_var_54',
'ico_var_55',
'ico_var_61',
'ico_var_62',
'if_var_66',
'if_var_67',
'if_var_74',
'if_var_78',
'if_var_79']

# %%
print ("STEP 3: DEVELOPING THE MODEL...")
X_train = df_train[in_model].drop(FILTER_EXCLUDE, errors='ignore', axis=1)
y_train = df_train[FILTER_TARGET]
X_test = df_test[in_model].drop(FILTER_EXCLUDE, errors='ignore', axis=1)

# %% 
## Submission ----
def prediction_submission_mfalonso(id, pred_test):
    df_test_tosend = pd.DataFrame({'id': id, 'pred': pred_test})
    filename = "df_test_tosend_GA_1_mfalonso.csv"
    df_test_tosend.to_csv(filename, sep=',')
    url = 'http://mfalonso.pythonanywhere.com/api/v1.0/uploadpredictions'
    files_ = {'file': (filename, open(filename, 'rb'))}

    submission_gini_score = None
    while submission_gini_score is None:
        try:
            rsub = requests.post(url, files=files_, auth=HTTPBasicAuth('juanbretti', '0e1a43f06b'))
            resp_str = str(rsub.text)
            submission_gini_score = float(resp_str.split(';')[1].split('=')[1])
            print("RESULT SUBMISSION (mfalonso):", submission_gini_score)  
            start_time = datetime.now()
        except:
            files_ = {'file': (filename, open(filename, 'rb'))}
            time.sleep(TIME_WAIT_RETRY)

    return submission_gini_score, start_time

def prediction_submission_manoelutad(id, pred_test):
    df_test_tosend = pd.DataFrame({'id': id, 'pred': pred_test})
    filename = "df_test_tosend_GA_1_manoelutad.csv"
    df_test_tosend.to_csv(filename, sep=',')
    url = 'http://manoelutad.pythonanywhere.com/uploadpredictions/6aQ6IxU7Va'
    files_ = {'file': (filename, open(filename, 'rb')),
            'pycode': (__file__, open(__file__, 'rb'))}

    submission_gini_score = None
    while submission_gini_score is None:
        try:
            rsub = requests.post(url, files=files_, auth=HTTPBasicAuth("juanbretti", "sha256$F7s0Yak4$c326c18e2afb65348d7c462ba09973e414380c36bab47c8439d3f193c76a3f94"))
            resp_str = str(rsub.text)
            submission_gini_score = float(resp_str.split(' = ')[1])
            print ("RESULT SUBMISSION (manoelutad):", submission_gini_score)  
            start_time = datetime.now()
        except:
            files_ = {'file': (filename, open(filename, 'rb')),
                      'pycode': (__file__, open(__file__, 'rb'))}
            time.sleep(TIME_WAIT_RETRY)

    return submission_gini_score, start_time

def test_name_model(start_time, model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    pred_test  = model.predict_proba(X_test)[:,1]
    while (datetime.now()-start_time).seconds < TIME_WAIT_SUBMIT:
        time.sleep(TIME_WAIT_RETRY)
    submission_gini_score, start_time = prediction_submission_manoelutad(range(0, len(pred_test)), pred_test)
    _, _ = prediction_submission_mfalonso(range(0, len(pred_test)), pred_test)
    return submission_gini_score, start_time

# %%
submission_results = pd.DataFrame()
start_time = datetime.now()-timedelta(seconds=TIME_WAIT_SUBMIT)

range_seed = [252]
range_n_estimators = [800]
range_bootstrap = [False]
range_class_weight = ['balanced_subsample']
range_min_weight_fraction_leaf = [0.0001]
range_strategy_fill_na = ['most_frequent']
range_criterion = ['gini']

range_total = list(itertools.product(range_strategy_fill_na, range_class_weight, range_criterion, range_seed, range_n_estimators, range_min_weight_fraction_leaf, range_bootstrap))
# random.seed(RANDOM_SEED)
random.shuffle(range_total, random.random)

for strategy_fill_na, class_weight, criterion, seed, n_estimators, min_weight_fraction_leaf, bootstrap in range_total:

    X_test = fill_na(X_test, strategy_fill_na)

    model = RandomForestClassifier(n_estimators=n_estimators, min_weight_fraction_leaf=min_weight_fraction_leaf, bootstrap=bootstrap, random_state=seed, n_jobs=-1, class_weight=class_weight, criterion=criterion)
    submission_gini, start_time = test_name_model(start_time, model, X_train, y_train, X_test)  

    # Store results
    submission_result = pd.DataFrame({
        'strategy_fill_na': [strategy_fill_na],
        'class_weight': [class_weight],
        'criterion': [criterion],
        'seed': [seed],
        'n_estimators': [n_estimators],
        'min_weight_fraction_leaf': [min_weight_fraction_leaf],
        'n_estimators': [n_estimators],
        'bootstrap': [bootstrap],
        'Gini': [submission_gini],
        'Submission time': [start_time],
        })
    print(submission_result.to_string(index=False))
    submission_results = submission_results.append(submission_result, ignore_index=True)
    submission_results.to_pickle('../Store/submission_results_grid_GA_1-temp.pkl')

print ("DONE, CHANGED SEEDS")

# %%
### Emergency restore ----
submission_results = pd.read_pickle('../Store/submission_results_grid_GA_1-temp.pkl')

### Results ----
submission_results = submission_results.sort_values(by='Gini', ascending=False)
timestr = time.strftime("%Y%m%d-%H%M%S")
submission_results.to_csv(f'../Store/submission_results_grid_GA-{timestr}.csv', sep=',')
submission_results

# %%