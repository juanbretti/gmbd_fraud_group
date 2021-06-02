# %% 
## Setup ----
### Libraries ----
print ("IMPORTING LIBRARIES...")
# General usage
import numpy as np
import pandas as pd
from datetime import datetime
from profmanoelgadi_support_package import (IV, PSI)
import profmanoelgadi_support_package as manoel

# Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import  GridSearchCV
from sklearn.metrics import make_scorer
import xgboost as xgb
from skopt import BayesSearchCV

# Reporting
from pandas_profiling import ProfileReport
from matplotlib import pyplot as plt

# Preprocessing
from sklearn.preprocessing import RobustScaler

# %%
### Constants ----
# Number of dimensions of the vector annoy is going to store. 
VECTOR_SIZE = 20
# Create reports and plots
FULL_EXECUTION = True
FULL_EXECUTION_REPORT = False
# Hyper parameter tunning
TUNNING_PARAM_COMB = 100
TUNNING_CV = 4

# %% 
## Loading data ----
print ("LOADING DATASETS...")
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
## Fill NA ----
print ("STEP 1: DOING MY TRANSFORMATIONS...")
df_train = df_train.fillna(0)
df_test = df_test.fillna(0)

# %%
## EDA ----
### Quick check ----
if FULL_EXECUTION_REPORT:
    pd.set_option('display.max_rows', 500)
    df_train.head(5).T

if FULL_EXECUTION_REPORT:
    df_train.nunique()
    df_test.nunique()

### ProfileReport ----
# Write an EDA report before any data change
if FULL_EXECUTION_REPORT:
    ProfileReport(df_train, title="Exploratory Data Analysis: Train: Raw", minimal=True).to_file("../EDA/df_train.html")
    ProfileReport(df_test, title="Exploratory Data Analysis: Test: Raw", minimal=True).to_file("../EDA/df_test.html")

# %%
## IV ----
# https://docs.tibco.com/pub/sfire-dsc/6.5.0/doc/html/TIB_sfire-dsc_user-guide/GUID-07A78308-525A-406F-8221-9281F4E9D7CF.html
# High IV is better
iv_summary, iv_var = manoel.IV.data_vars(df_train.drop(['ob_target'], axis=1), df_train['ob_target'])

## PSI ----
# https://www.listendata.com/2015/05/population-stability-index.html
# http://ucanalytics.com/blogs/population-stability-index-psi-banking-case-study/
# Low PSI is better
PSI_list = []
for item in df_train.drop(['ob_target'], axis=1).columns:
    PSI_list.append(manoel.PSI.calculate_psi(df_train[item], df_test[item], buckettype='bins', number=10))

iv_var['PSI'] = PSI_list

# %%
iv_psi_var = iv_var.sort_values('IV', ascending=False)
iv_psi_var = iv_psi_var[(iv_psi_var['PSI']<=0.25) & (iv_psi_var['IV']>=0.02)]

# %%
## Columns transformation ----
### Filters ----
filter_ib = df_train.filter(regex='ib_var').columns
filter_icn = df_train.filter(regex='icn_var').columns
filter_ico = df_train.filter(regex='ico_var').columns
filter_if = df_train.filter(regex='if_var').columns
filter_exclude = ['Unnamed: 0', 'id', 'contract_date']
filter_include_iv_psi = iv_psi_var['VAR_NAME'].tolist()
filter_cat_diff = ['icn_var_22', 'icn_var_24', 'ico_var_26', 'ico_var_27', 'ico_var_33', 'ico_var_34', 'ico_var_35', 'ico_var_36', 'ico_var_37']
filter_target = 'ob_target'

### Transformations ----
def scaler_transform(df, encoder=None):
    if encoder is None:
        encoder = RobustScaler()
        encoder.fit(df)
    df_encoded = encoder.transform(df)
    df_encoded = pd.DataFrame(df_encoded, columns = df.columns)
    return df_encoded, encoder

def df_transform(df, enc_scaler_if_ico=None, enc_scaler_cat_diff=None):
    #ib
    filter_ = filter_ib.intersection(filter_include_iv_psi).drop(filter_exclude+filter_cat_diff, errors='ignore')
    df_ib = df[filter_ib]

    # icn
    filter_ = filter_icn.intersection(filter_include_iv_psi).drop(filter_exclude+filter_cat_diff, errors='ignore')
    if len(filter_)>0:
        data = df[filter_]
        df_icn = pd.get_dummies(data, columns = data.columns, drop_first=True)
    else:
        df_icn = pd.DataFrame()

    # if + ico
    filter_ = filter_if.append(filter_ico).intersection(filter_include_iv_psi).drop(filter_exclude+filter_cat_diff, errors='ignore')
    data = df[filter_]
    if enc_scaler_if_ico is None:
        df_if_ico, enc_scaler_if_ico = scaler_transform(data, None)
    else:
        df_if_ico, _ = scaler_transform(data, enc_scaler_if_ico)

    # cat_diff
    filter_ = list(set(filter_cat_diff).intersection(filter_include_iv_psi).difference(filter_exclude))
    data = df[filter_]
    if enc_scaler_cat_diff is None:
        df_cat_diff, enc_scaler_cat_diff = scaler_transform(data, None)
    else:
        df_cat_diff, _ = scaler_transform(data, enc_scaler_cat_diff)

    df_train_transformed = pd.concat([df_ib, df_icn, df_if_ico, df_cat_diff], axis=1)

    return df_train_transformed, enc_scaler_if_ico, enc_scaler_cat_diff

df_train_transformed, enc_scaler_if_ico, df_train_enc_scaler_cat_diff = df_transform(df_train, None, None)
df_test_transformed, _, _ = df_transform(df_test, enc_scaler_if_ico, df_train_enc_scaler_cat_diff)

# %% 
## Data split ----
X_train = df_train_transformed
y_train = df_train[filter_target]
X_test = df_test_transformed

# %%
## SMOTE ----
# https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html#imblearn.over_sampling.SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42, n_jobs=-1)
X_train, y_train = smote.fit_resample(X_train, y_train)
# y_train.value_counts()
# y_train_SMOTE.value_counts()

# %%
## Auxiliary functions ----
def timer(start_time=None):
    """Create a 'timer' object to measure execution time 

    Args:
        start_time (datetime[64], optional): End time when set. Defaults to None.

    Returns:
        str: Time elapsed since execution
    """
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

# %%
## RamdomForest ----
### Default ----
clf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced_subsample')
clf.fit(X_train, y_train)
pred_train = clf.predict_proba(X_train)[:,1]
pred_test  = clf.predict_proba(X_test)[:,1]

##### Feature importance ----
# https://mljar.com/blog/feature-importance-in-random-forest/
if FULL_EXECUTION_REPORT:
    plt.rcParams['figure.figsize'] = [15, 30]
    sorted_idx = clf.feature_importances_.argsort()
    plt.barh(X_train.columns[sorted_idx], clf.feature_importances_[sorted_idx])

# df_describe = pd.DataFrame(clf.feature_importances_)
# df_describe.describe()

# %%
### Filter importance ----
features_important = (clf.feature_importances_>0.001).tolist()
pd.Series(features_important).value_counts()

clf2 = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced_subsample')
clf2.fit(X_train.iloc[:, features_important], y_train)
pred_train = clf2.predict_proba(X_train.iloc[:, features_important])[:,1]
pred_test  = clf2.predict_proba(X_test.iloc[:, features_important])[:,1]

##### Feature importance ----
if FULL_EXECUTION_REPORT:
    plt.rcParams['figure.figsize'] = [15, 30]
    sorted_idx = clf2.feature_importances_.argsort()
    plt.barh(X_train.columns[sorted_idx], clf2.feature_importances_[sorted_idx])

# %%
### Hyperparameter tunning ----
# https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
params = {
    'n_estimators': [10, 100, 1000],
    'max_depth': [None, 10, 50, 100],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': ['auto', None]
    }

rf_model_grid = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced_subsample')
rf_model_search = GridSearchCV(rf_model_grid, param_grid=params, scoring='roc_auc', n_jobs=-1, verbose=3)

start_time = timer(None) # timing starts from this point for "start_time" variable
rf_model_search.fit(X_train, y_train)
timer(start_time) # timing ends here for "start_time" variable

pred_train = rf_model_search.predict_proba(X_train)[:,1]
pred_test  = rf_model_search.predict_proba(X_test)[:,1]

# %%
### Custom scorer ----
# https://stackoverflow.com/a/50380029/3780957
# https://campus.ie.edu/webapps/discussionboard/do/message?action=list_messages&course_id=_114365970_1&nav=discussion_board_entry&conf_id=_270898_1&forum_id=_136661_1&message_id=_5285111_1

def fraud_cost_i(y, y_pred):
    # $cost = $100 x fn + $10 x fp + $1 x tp
    # FN
    if (y == 1) & (y_pred == 0):
        cost = 100
    # FP
    elif (y == 0) & (y_pred == 1):
        cost = 10
    # TP
    elif (y == 1) & (y_pred == 1):
        cost = 1
    # TN
    elif (y == 0) & (y_pred == 0):
        cost = 1
    else:
        cost = 0
    return cost

def fraud_cost(y, y_pred):
    cost = list(map(lambda a, b: fraud_cost_i(a, b), y, y_pred))
    return sum(cost)

custom_scorer = make_scorer(fraud_cost, greater_is_better=False, needs_proba=False)

params = {
    'n_estimators': [10, 100, 1000],
    'max_depth': [None, 10, 50, 100],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': ['auto', None]
    }

rf_model_grid = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced_subsample')
rf_model_search = GridSearchCV(rf_model_grid, param_grid=params, scoring=custom_scorer, n_jobs=-1, verbose=3)

start_time = timer(None) # timing starts from this point for "start_time" variable
rf_model_search.fit(X_train, y_train)
timer(start_time) # timing ends here for "start_time" variable

pred_train = rf_model_search.predict_proba(X_train)[:,1]
pred_test  = rf_model_search.predict_proba(X_test)[:,1]

# %%
#### Feature importance ----
# https://mljar.com/blog/feature-importance-in-random-forest/
if FULL_EXECUTION_REPORT:
    plt.rcParams['figure.figsize'] = [15, 30]
    sorted_idx = rf_model_search.best_estimator_.feature_importances_.argsort()
    plt.barh(X_train.columns[sorted_idx], rf_model_search.best_estimator_.feature_importances_[sorted_idx])

# %%
## XGBoost: Hyperparameter tunning ----
params = {
    'max_depth': list(range(3,10,2)),
    'min_child_weight': list(range(1,6,2)),
    'gamma': [i/10.0 for i in range(0,5)],
    'subsample': [i/10.0 for i in range(6,10)],
    'colsample_bytree':[i/10.0 for i in range(6,10)],
    'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
}

# Build the model
xgb_model_bayes = xgb.XGBClassifier(
    n_estimators=10,
    objective ='binary:logistic',
    tree_method='gpu_hist', gpu_id=0, nthread=-1,
    random_state=42)

fitted_model = BayesSearchCV(xgb_model_bayes, search_spaces=params, n_iter=TUNNING_PARAM_COMB, scoring='accuracy', n_jobs=-1, verbose=3, random_state=42)
fitted_model.fit(X_train, y_train)
pred_train = fitted_model.predict_proba(X_train)[:,1]
pred_test  = fitted_model.predict_proba(X_test)[:,1]

# %% 
## Score ----
print ("STEP 4: ASSESSING THE MODEL...")
# CALCULATING GINI PERFORMANCE ON DEVELOPMENT SAMPLE
# Watch this video for reference: https://youtu.be/MiBUBVUC8kE
from sklearn.metrics import roc_auc_score
gini_score = 2*roc_auc_score(y_train, pred_train)-1
print ("GINI DEVELOPMENT=", gini_score)

# %% 
## Submit ----
# http://manoelutad.pythonanywhere.com/ranking/6aQ6IxU7Va
print ("STEP 5: SUBMITTING THE RESULTS...  DO NOT CHANGE THIS PART!")
import requests
from requests.auth import HTTPBasicAuth
df_test['pred'] = pred_test
df_test['id'] = df_test.iloc[:,0]
df_test_tosend = df_test[['id','pred']]

filename = "df_test_tosend.csv"
df_test_tosend.to_csv(filename, sep=',')
url = 'http://manoelutad.pythonanywhere.com/uploadpredictions/6aQ6IxU7Va'
files = {'file': (filename, open(filename, 'rb')),
         'pycode': (__file__, open(__file__, 'rb'))}

#rsub = requests.post(url, files=files)
rsub = requests.post(url, files=files, auth=HTTPBasicAuth("juanbretti", "sha256$F7s0Yak4$c326c18e2afb65348d7c462ba09973e414380c36bab47c8439d3f193c76a3f94"))
resp_str = str(rsub.text)
print ("RESULT SUBMISSION: ", resp_str)

# %%
