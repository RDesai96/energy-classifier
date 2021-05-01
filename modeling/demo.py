import pandas as pd
import os
import gradio as gr
import pickle

# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
# from sklearn.ensemble import RandomForestClassifier
# import pickle
#
#
# df_train = pd.read_csv('data/cleandata/Train.csv')
# df_test = pd.read_csv('data/cleandata/Test.csv')
#
# X_train = df_train.drop('Power Source',axis=1)
# y_train  = df_train['Power Source']
# X_test = df_test.drop('Power Source',axis=1)
# y_test  = df_test['Power Source']
#
# feat_names = X_train.columns
#
# imputer = IterativeImputer(sample_posterior=True,
#                            imputation_order='descending',
#                            skip_complete=True,
#                            random_state=42)
#
# scaler = StandardScaler()
#
#
# rf = RandomForestClassifier(n_estimators=500, criterion='gini',
#                             max_features='sqrt', oob_score=True,
#                             random_state=42)
#
# pipeRF = Pipeline( [  ('impute',  imputer),
#                       ('rf',     rf ) ])
#
#
# pipeRF.fit(X_train,y_train)
# filename = 'modeling/RF_model.sav'
# pickle.dump(pipeRF, open(filename, 'wb'))
os.system("mkdir RF")
os.system("cd RF; gdown https://drive.google.com/uc?id=1wwR5BKVG4L4jy1_N805Wo9Q99Yu5EiVt -O RF_model.sav; cd ..")
loaded_model = pickle.load(open('RF/RF_model.sav', 'rb'))



def Farm_Predictor(DNI, GHI, a40, a35, a30, Capacity, Longitude, Latitude):
    df = pd.DataFrame.from_dict({'DNI':[DNI], 'GHI':[GHI], 'a40':[a40],
                                 'a35':[a35], 'a30':[a30], 'Capacity':[Capacity],
                                 'Longitude':[Longitude], 'Latitude': [Latitude]})
    pred = loaded_model.predict_proba(df)[0]
    return {'Solar': pred[0], 'Wind': pred[1]}

Source = gr.inputs.Radio(['Solar','Wind'], label='Power Source')
DNI = gr.inputs.Slider(minimum=3, maximum=9, default=4, label="DNI")
GHI = gr.inputs.Slider(minimum=3, maximum=6.5, default=4, label="GHI")
a40 = gr.inputs.Slider(minimum=0, maximum=425, default=100, label="a40 Wind Potential (MW)")
a35 = gr.inputs.Slider(minimum=0, maximum=425, default=100, label="a35 Wind Potential (MW)")
a30 = gr.inputs.Slider(minimum=0, maximum=425, default=100, label="a30 Wind Potential (MW)")
Capacity = gr.inputs.Slider(minimum=0.05, maximum=2500, default=100, label="Capacity (MW)")
Longitude = gr.inputs.Slider(minimum=-130, maximum=-60, default=-90, label="Longitude")
Latitude = gr.inputs.Slider(minimum=20, maximum=50, default=30, label="Latitude")


gr.Interface(Farm_Predictor, [DNI, GHI, a40, a35, a30, Capacity, Longitude, Latitude],
             "label", live=True).launch()