###########
#
# FEW DAYS BACK - IN THE CHALLENGE CIFF.DDNS.NET - YOU SHOULD HAVE RECEIVED A HINT ABOUT IMPLEMENTING A GENETIC ALGORITHM.
#
# SHOULD YOU WANT TO DO IT, THIS BASE CODE SHOULD HELP YOU TO IMPLEMENT GENETIC ALGORITHM FOR FEATURE SELECTION! (SYNONIMOUS= FEATURES, CHARACTERISTICS AND VARIABLES)
#
# THIS CODE NEEDS TO BE ADAPTED TO YOUR CODE!!! 
# NOTE THAT IT USES LOGISTIC REGRESSION JUST FOR TESTING PURPOSE, CHANGE TO THE METHOD YOU ARE USING AND EVERYHTING ELSE YOU HAVE DONE AS WELL.
#
# QUESTIONS TO: manoelgadi@campusciff.net
#
#############
# %%
## IMPORTING LIBRARIES ----
print("IMPORTING LIBRARIES...")
from numpy.core.fromnumeric import var
import pandas as pd
import numpy as np
import statsmodels.api as sm
import re
from deap import creator, base, tools, algorithms #GENETIC ALGORITHM LIBRARY - requirement: pip install deap
import random
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import time
from datetime import datetime, timedelta
from sklearn.impute import SimpleImputer
import requests
from requests.auth import HTTPBasicAuth
import pickle

# %%
## DOWNLOADING DATASETS ----
print("DOWNLOADING DATASETS...")
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

### Constants ----
RANDOM_STATE = 252
N_ESTIMATORS = 1000
BOOTSTRAP = False
CLASS_WEIGHT = 'balanced_subsample'
MIN_WEIGHT_FRACTION_LEAF = 0.0001
FILL_NA_STRATEGY = 'most_frequent'

FILTER_EXCLUDE = ['Unnamed: 0', 'id', 'contract_date', 'ib_var_12']
FILTER_TARGET = 'ob_target'
TIME_WAIT_SUBMIT = 25
TIME_WAIT_RETRY = 1

### Drop ----
df = df_train.drop(FILTER_EXCLUDE, errors='ignore', axis=1) #DEV-SAMPLE
dfo = df_test.drop(FILTER_EXCLUDE, errors='ignore', axis=1) #OUT-OF-TIME SAMPLE

### Random ----
import random
random.seed(RANDOM_STATE)

# %%
## IDENTIFYING TYPES ----
print ("IDENTIFYING TYPES...")
in_model = []
list_ib = list()  #input binary
list_icn = list() #input categorical nominal
list_ico = list() #input categorical ordinal
list_if = list()  #input numerical continuos (input float)
list_inputs = list()
output_var = 'ob_target'

for var_name in df.columns:
    if re.search('^ib_',var_name):
        list_inputs.append(var_name)      
        list_ib.append(var_name)
    elif re.search('^icn_',var_name):
        list_inputs.append(var_name)      
        list_icn.append(var_name)
    elif re.search('^ico_',var_name):
        list_inputs.append(var_name)      
        list_ico.append(var_name)
    elif re.search('^if_',var_name):
        list_inputs.append(var_name)      
        list_if.append(var_name)
    elif re.search('^ob_',var_name):
        output_var = var_name
    else:
        print ("ERROR: unable to identify the type of:", var_name)

# %% 
## Submission ----
def prediction_submission_mfalonso(id, pred_test):
    df_test_tosend = pd.DataFrame({'id': id, 'pred': pred_test})
    filename = "df_test_tosend_GA_5_mfalonso.csv"
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
    filename = "df_test_tosend_GA_5_manoelutad.csv"
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
    submission_gini_score_manoelutad, start_time = prediction_submission_manoelutad(range(0, len(pred_test)), pred_test)
    submission_gini_score_mfalonso, _ = prediction_submission_mfalonso(range(0, len(pred_test)), pred_test)
    return submission_gini_score_manoelutad, submission_gini_score_mfalonso, start_time

def fill_na(df_test, strategy='most_frequent'):
    if strategy == 'most_frequent':
        test_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    else:
        test_imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=strategy)
    df_filled = test_imputer.fit_transform(df_test)
    df_filled = pd.DataFrame(df_filled, columns=df_test.columns)
    return df_filled

def store_advace(add_time=False):
    if add_time:
        timestr = time.strftime("%Y%m%d-%H%M%S")
    else:
        timestr = 'Generic'

    try:
        path_to_save = f'../Store/5/GA_var_5_population_{RANDOM_STATE}-{timestr}.pkl'
        with open(path_to_save, 'wb') as f:
            pickle.dump(population, f)
    except:
        pass
    try:    
        path_to_save = f'../Store/5/GA_var_5_dic_gini_{RANDOM_STATE}-{timestr}.pkl'
        with open(path_to_save, 'wb') as f:
            pickle.dump(dic_gini, f)
    except:
        pass
    try:    
        path_to_save = f'../Store/5/GA_var_5_list_gini_{RANDOM_STATE}-{timestr}.pkl'
        with open(path_to_save, 'wb') as f:
            pickle.dump(list_gini, f)
    except:
        pass

# %%
## GENETIC ALGORITHM FOR FEATURE SELECTION ----
print("GENETIC ALGORITHM FOR FEATURE SELECTION:")

#####
#SETING UP THE GENETIC ALGORITHM and CALCULATING STARTING POOL (STARTING CANDIDATE POPULATION)
#####
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(list_inputs))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
def evalOneMax(individual):
    return sum(individual),

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

NPOPSIZE = 50 #RANDOM STARTING POOL SIZE
population = toolbox.population(n=NPOPSIZE)

# %%
#####
## ASSESSING GINI ON THE STARTING POOL ----
#####
start_time = datetime.now()-timedelta(seconds=TIME_WAIT_SUBMIT)
submission_results = pd.DataFrame()
dic_gini={}

for i in range(np.shape(population)[0]): 

    print('ASSESSING GINI ON THE STARTING POOL', i)
    # TRASLATING DNA INTO LIST OF VARIABLES (1-81)
    var_model = []    
    for j in range(np.shape(population)[1]): 
        if (population[i])[j]==1:
            var_model.append(list_inputs[j])

    # ASSESSING GINI INDEX FOR EACH INVIVIDUAL IN THE INITIAL POOL 
            
    X_train=df[var_model]
    Y_train=df[output_var]
    X_test = fill_na(dfo[var_model], FILL_NA_STRATEGY)

    ######
    # CHANGE_HERE - START: YOU ARE VERY LIKELY USING A DIFFERENT TECHNIQUE BY NOW. SO CHANGE TO YOURS.
    #####
    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, min_weight_fraction_leaf=MIN_WEIGHT_FRACTION_LEAF, bootstrap=BOOTSTRAP, random_state=RANDOM_STATE, n_jobs=-1, class_weight=CLASS_WEIGHT)
    submission_gini_score_manoelutad, submission_gini_score_mfalonso, start_time = test_name_model(start_time, model, X_train, Y_train, X_test)
    
    store_advace()        
    submission_result = pd.DataFrame({
        'manoelutad': [submission_gini_score_manoelutad],
        'mfalonso': [submission_gini_score_mfalonso],
        'var_model': [var_model],
        'vector': [str(population[i])],
        'time': [start_time],
        'source': ['ASSESSING GINI ON THE STARTING POOL'],
        'i': [i],
    })
    submission_results = submission_results.append(submission_result, ignore_index=True)
    submission_results.to_pickle('../Store/5/submission_results_GA-temp.pkl')
    ######
    # CHANGE_HERE - END: HERE IT USES THE DEVELOPMENT GINI TO SELECT VARIABLES, YOU SHOULD A DIFFERENT GINI. EITHER THE OOT GINI OR THE SQRT(DEV_GINI*OOT_GINI)
    #####                
    
    gini=str(submission_gini_score_manoelutad)+";"+str(population[i]).replace('[','').replace(', ','').replace(']','')
    dic_gini[gini]=population[i]   
list_gini=sorted(dic_gini.keys(),reverse=True)

store_advace(True)

# %%
#####
## GENETIC ALGORITHM MAIN LOOP - START ----
# - ITERATING MANY TIMES UNTIL NO IMPROVMENT HAPPENS IN ORDER TO FIND THE OPTIMAL SET OF CHARACTERISTICS (VARIABLES)
#####
sum_current_gini=0.0
sum_current_gini_1=0.0
sum_current_gini_2=0.0
first=0    
OK = 1
a=0

while OK:  #REPEAT UNTIL IT DO NOT IMPROVE, AT LEAST A LITLE, THE GINI IN 2 GENERATIONS
    
    print('GENETIC ALGORITHM MAIN LOOP', a)

    a=a+1
    print('loop ', a)
    OK=0

    ####
    # GENERATING OFFSPRING - START
    ####
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1) #CROSS-X PROBABILITY = 50%, MUTATION PROBABILITY=10%
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population =toolbox.select(offspring, k=len(population))
    ####
    # GENERATING OFFSPRING - END
    ####

    sum_current_gini_2=sum_current_gini_1
    sum_current_gini_1=sum_current_gini
    sum_current_gini=0.0

    #####
    #ASSESSING GINI ON THE OFFSPRING - START
    #####
    for j in range(np.shape(population)[0]): 
        if population[j] not in dic_gini.values(): 
            var_model = [] 
            for i in range(np.shape(population)[1]): 
                if (population[j])[i]==1:
                    var_model.append(list_inputs[i])
            
            X_train=df[var_model]
            Y_train=df[output_var]
            X_test = fill_na(dfo[var_model], FILL_NA_STRATEGY)
            
            ######
            # CHANGE_HERE - START: YOU ARE VERY LIKELY USING A DIFFERENT TECHNIQUE BY NOW. SO CHANGE TO YOURS.
            #####            
            model = RandomForestClassifier(n_estimators=N_ESTIMATORS, min_weight_fraction_leaf=MIN_WEIGHT_FRACTION_LEAF, bootstrap=BOOTSTRAP, random_state=RANDOM_STATE, n_jobs=-1, class_weight=CLASS_WEIGHT)
            submission_gini_score_manoelutad, submission_gini_score_mfalonso, start_time = test_name_model(start_time, model, X_train, Y_train, X_test)
            
            store_advace()        
            submission_result = pd.DataFrame({
                'manoelutad': [submission_gini_score_manoelutad],
                'mfalonso': [submission_gini_score_mfalonso],
                'var_model': [var_model],
                'vector': [str(population[j])],
                'time': [start_time],
                'source': ['ASSESSING GINI ON THE OFFSPRING'],
                'a': [a],
                'j': [j],
            })
            submission_results = submission_results.append(submission_result, ignore_index=True)
            submission_results.to_pickle('../Store/5/submission_results_GA-temp.pkl')
            ######
            # CHANGE_HERE - END: HERE IT USES THE DEVELOPMENT GINI TO SELECT VARIABLES, YOU SHOULD A DIFFERENT GINI. EITHER THE OOT GINI OR THE SQRT(DEV_GINI*OOT_GINI)
            #####                       
           
            gini=str(submission_gini_score_manoelutad)+";"+str(population[j]).replace('[','').replace(', ','').replace(']','')
            dic_gini[gini]=population[j]  
    #####
    #ASSESSING GINI ON THE OFFSPRING - END
    #####

    #####
    #SELECTING THE BEST FITTED AMONG ALL EVER CREATED POPULATION AND CURRENT OFFSPRING - START
    #####           
    list_gini=sorted(dic_gini.keys(),reverse=True)
    population=[]
    for i in list_gini[:NPOPSIZE]:
        population.append(dic_gini[i])
        gini=float(i.split(';')[0])
        sum_current_gini+=gini
    #####
    #SELECTING THE BEST FITTED AMONG ALL EVER CREATED POPULATION AND CURRENT OFFSPRING - END
    #####           
      
    #HAS IT IMPROVED AT LEAST A LITLE THE GINI IN THE LAST 2 GENERATIONS
    print('sum_current_gini=', sum_current_gini, 'sum_current_gini_1=', sum_current_gini_1, 'sum_current_gini_2=', sum_current_gini_2)
    if(sum_current_gini>sum_current_gini_1+0.0001 or sum_current_gini>sum_current_gini_2+0.0001):
        OK=1
#####
#GENETIC ALGORITHM MAIN LOOP - END
#####

store_advace(True)

gini_max=list_gini[0]        
gini=float(gini_max.split(';')[0])
features=gini_max.split(';')[1]

# %%
####
## PRINTING OUT THE LIST OF FEATURES ----
#####
feature_=0
feature_list = []
for i in range(len(features)):
    if features[i]=='1':
        feature_+=1
        print('feature ', feature_, ':', list_inputs[i])
        feature_list.append(list_inputs[i])
print('gini: ', gini)

store_advace(True)

# %%
