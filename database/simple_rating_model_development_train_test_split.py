# %%
# Import libraries
import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# %%
#... READING ALL DATA FRAMES

big_df = pd.DataFrame() 
    
for filename in glob.glob("./Rating_*.xlsx"):
#    connres.commit()
    print ("READING DATABASE: "+filename)
    df = pd.read_excel(open(filename,'rb'), sheet_name="Resultados",header = None) #Reading SABI Export without index  
    df.columns = ['id', 'nombre_x', 'nif', 'nombre', 'provincia', 'calle', 'telefono', 'web', 'desc_actividad', 'cnae', 'cod_consolidacion', 'rating_grade_h2', 'rating_grade_h1', 'rating_grade_h0', 'rating_numerico_h2', 'rating_numerico_h1', 'rating_numerico_h0', 'modelo_propension_h2', 'modelo_propension_h1', 'modelo_propension_h0', 'guo_nombre', 'guo_id_bvd', 'guo_pais', 'guo_tipo', 'estado_detallado', 'fecha_cambio_estado', 'fecha_constitucion', 'p10000_h0', 'p10000_h1', 'p10000_h2', 'p20000_h0', 'p20000_h1', 'p20000_h2', 'p31200_h0', 'p31200_h1', 'p31200_h2', 'p32300_h0', 'p32300_h1', 'p32300_h2', 'p40100_mas_40500_h0', 'p40100_mas_40500_h1', 'p40100_mas_40500_h2', 'p40800_h0', 'p40800_h1', 'p40800_h2', 'p49100_h0', 'p49100_h1', 'p49100_h2']
    df['h0_anio'] = 2017     
    df = df.fillna('')
    df=df.drop(df.index[0]) #Dropping SABI variable names.
    df['nif'] = df.nif.str.upper() #CONVERTING cif INTO UPPERCASE
    for partida in ['p10000_h0', 'p10000_h1', 'p10000_h2', 'p20000_h0', 'p20000_h1', 'p20000_h2', 'p31200_h0', 'p31200_h1', 'p31200_h2', 'p32300_h0', 'p32300_h1', 'p32300_h2', 'p40100_mas_40500_h0', 'p40100_mas_40500_h1', 'p40100_mas_40500_h2', 'p40800_h0', 'p40800_h1', 'p40800_h2', 'p49100_h0', 'p49100_h1', 'p49100_h2']:
#        print (partida,"ha sido convertido en numerico")
        df[partida] = pd.to_numeric(df[partida], errors='coerce').fillna(0)- 0.005
    df['nif_normalizado'] = df['nif'].str[-8:]    
    big_df = big_df.append(df, ignore_index=True)     

# %%
# Define target variable
df = big_df
df['target_status'] = [0 if i in ['Activa', ''] else 1 for i in df['estado_detallado']] # 0 si Activa, 1 si algo raro!

# Calculate dependent variables

# _h0, _h1, _h2
# _h0: history 0, here h0 means the year 2017 (historia 0, aquí h0 significa el año 2017)
# _h1: history -1, here h1 means the year 2016 (historia -1, aquí h1 significa el año 2016)
# _h2: history -2, here h2 means the year 2015 (historia -2, aquí h2 significa el año 2015)

# Ebita Margin - Ebitda / Turn over (Ventas)
# p49100: Profit (Resultado del ejercicio)
# p40800: Amortization (Amortización) 
# p40100: Sales Turnover (Ingresos de Explotación) >> Revenue
# p40500: Other sales (Otros Ingresos)
df['ebitda_income'] = (df.p49100_h1+df.p40800_h1)/(df.p40100_mas_40500_h1) 

# Total Debt / Ebitada
# p31200: Short Term Debt / Deuda a corto plazo
# p32300: Long Term Debt / Deuda a largo plazo
# p49100: Profit (Resultado del ejercicio) >> Income
# p40800: Amortization (Amortización) 
df['debt_ebitda'] =(df.p31200_h1 + df.p32300_h1) /(df.p49100_h1+df.p40800_h1) 

# rraa_rrpp: Financial leveraging / apalancamiento financiero 
# p10000: Total Assets / Total activos
# p20000: Own Capital / Patrimonio neto
df['rraa_rrpp'] = (df.p10000_h1 - df.p20000_h1) /df.p20000_h1

# Log of Operating Income
df['log_operating_income'] = np.log(df.p40100_mas_40500_h1)

# Additional requirement
# easy option) include a new ratio on the development code, fit a new model and dump it and load it in the app.py (again, this is not graded)

# EBITDA / Total Assets
# EBITDA Return on Assets
# https://www.arborinvestmentplanner.com/return-on-total-assets-ratios-calculations/
df['return_on_assets'] = (df.p49100_h1+df.p40800_h1) / df.p10000_h1

# %%
# Additional requirement
# hard option) Using the first digit of CNAE as industry, apply standard scaler transformation by industry, fit a new model, dump (export) the new fitted model and also dump the standard scale transformations. Implement the new model and the standard scale transformations by industry into app.py (again, this is not graded)

# Create the CNAE
df['cnae_first'] = df['cnae'].astype(str).str[0]
df = df[~df['cnae_first'].isna()]

# Define columns to use in the model
columns_to_scale = ['ebitda_income','debt_ebitda','rraa_rrpp','log_operating_income', 'return_on_assets']
column_target = 'target_status'

# Split the dataset
df_train, df_test = train_test_split(df, test_size=0.1, random_state=42, stratify=df[column_target])

# %%
df_concat = pd.DataFrame()
scaler_concat = {}

# Calculates a scaler 
# Scaler helps to normalize the dependent variables between the different industry type `CNAE`
# The script creates one scaler per first character of the `CNAE` code
for first in df_train['cnae_first'].unique():
    # Select X
    df_first = df_train[df_train['cnae_first']==first]
    df_first = df_first[columns_to_scale+[column_target]].replace([np.inf, -np.inf], np.nan).dropna()
    # Scale X
    encoder = StandardScaler().fit(df_first[columns_to_scale])
    df_first_scaled = encoder.transform(df_first[columns_to_scale])
    df_first_scaled = pd.DataFrame(df_first_scaled, columns=columns_to_scale)
    # Add y
    df_first_scaled[column_target] = df_first[column_target].to_list()
    # Concat
    df_concat = df_concat.append(df_first_scaled)
    scaler_concat.update({first: encoder})

# %%
# Create the classifier model
model = RandomForestClassifier(random_state=42, bootstrap=False, class_weight='balanced_subsample', n_estimators=100, n_jobs=-1)

# Train
X_train = df_concat[columns_to_scale]
y_train = df_concat[column_target]

fitted_model = model.fit(X_train, y_train)

y_train_pred = fitted_model.predict(X_train)
y_train_pred_proba = fitted_model.predict_proba(X_train)[:,1]

# %%
# Test
df_first = df_test[columns_to_scale+[column_target]+['cnae_first']]
df_first = df_first.replace([np.inf, -np.inf], np.nan).dropna()  # This could be improved

# Apply the `scaler` based on the `cnae_first`
X_test = pd.DataFrame()
for idx, row in df_first.iterrows():
    cnae_company = row['cnae_first']

    X = row[columns_to_scale].to_frame().T
    row_scaled = scaler_concat[cnae_company].transform(X)
    
    X_test = X_test.append(pd.DataFrame(row_scaled, columns=columns_to_scale), ignore_index=True)

y_test = df_first[column_target]

y_test_pred = fitted_model.predict(X_test)
y_test_pred_proba = fitted_model.predict_proba(X_test)
y_test_pred_proba_true = y_test_pred_proba[:,1]

# %%
print ("ASSESSING THE MODEL...")
# CALCULATING GINI PERFORMANCE ON DEVELOPMENT SAMPLE
from sklearn.metrics import roc_auc_score
gini_score = 2*roc_auc_score(y_test, y_test_pred_proba_true)-1
print ("GINI DEVELOPMENT=", gini_score)

from sklearn.metrics import accuracy_score
print("Accuracy: {0}".format(accuracy_score(y_test,y_test_pred)))

from sklearn.metrics import confusion_matrix
print('Confusion matrix:\n', confusion_matrix(y_test, y_test_pred))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_test_pred))

# %%
# https://scikit-plot.readthedocs.io/en/stable/metrics.html#scikitplot.metrics.plot_roc
# https://towardsdatascience.com/optimal-threshold-for-imbalanced-classification-5884e870c293
import matplotlib.pyplot as plt
import scikitplot as skplt
skplt.metrics.plot_roc(y_test, y_test_pred_proba)
plt.axvline(x=0.4)
plt.show()

# %%
print ("SAVING THE PERSISTENT MODEL...")
from joblib import dump, load

dump(fitted_model, 'Rating_RandomForestClassifier.joblib') 
dump(scaler_concat, 'scaler_concat.joblib') 
dump({'y': y_test,  'y_pred_proba': y_test_pred_proba}, 'y_values.joblib') 

# y_values = load('y_values.joblib') 
# y_values['y'].to_list()
# y_values['y_pred_proba']

# %%
# Calculate the cost per approval
# https://stackoverflow.com/a/50380029/3780957
# https://campus.ie.edu/webapps/discussionboard/do/message?action=list_messages&course_id=_114365970_1&nav=discussion_board_entry&conf_id=_270898_1&forum_id=_136661_1&message_id=_5285111_1

def approval_cost_i(y, y_pred_proba, threshold=0.5):
    # FN
    if (y == 1) & (y_pred_proba < threshold):
        cost = 100 # I will loose the interests from the operation
    # FP
    elif (y == 0) & (y_pred_proba >= threshold):
        cost = 10 # I could loose the loaned capital, the insurance will kick in. $10 is the cost of the insurance.
    # TP
    elif (y == 1) & (y_pred_proba >= threshold):
        cost = 1
    # TN
    elif (y == 0) & (y_pred_proba < threshold):
        cost = 1
    else:
        cost = 0
    return cost

def approval_cost(threshold):
    return sum(map(approval_cost_i, y_test, y_test_pred_proba_true, [threshold]*len(y_test)))

space_threshold = [10**x for x in np.linspace(-10,0,100)]
df_space = pd.DataFrame({'Threshold': space_threshold,
                         'Cost': [approval_cost(x) for x in space_threshold]})

df_space.plot(x='Threshold', y='Cost')
# %%
# Minimum value of the cost, to define the threshold
df_space[df_space['Cost'] == min(df_space['Cost'])]

# %%