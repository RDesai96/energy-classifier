import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import gradio as gr

df = pd.read_csv('data/cleandata/MergedComplete.csv')

df[['State', 'Power Source']] = df[['State', 'Power Source']].astype('category')
df_learn = df.drop(['Power Source', 'State'] ,axis=1)
feat_names = df_learn.columns

X_train, X_test, y_train, y_test = \
    train_test_split(df_learn, df['Power Source'],
                     test_size=0.3,
                     random_state=42)

knn_impute = KNNImputer()
scaler = StandardScaler()

# TODO: Trees/Bagging
# TODO: Plots (ROC curves, variable importance), check for well calibrated probabilities

# Logistic Elastic Net
# log_elastic_net = LogisticRegressionCV(cv=5,
#                                        penalty='elasticnet',
#                                        solver='saga',
#                                        random_state=42,
#                                        max_iter=10000,
#                                        l1_ratios=[0,0.2,0.4,0.6,0.8,.9,1] )
#
# pipe = Pipeline( [  ('impute',  knn_impute),
#                     ('scale',   scaler),
#                     ('elasticnet', log_elastic_net ) ])
#
# pipe.fit(X_train,y_train)
# coefs = pd.DataFrame(pipe[2].coef_, columns=df_learn.columns)
# plt.bar(x=feat_names, height=abs(coefs.values.flatten())/(abs(coefs.values).sum()))
# plt.title('Logistic Elastic Net Feature Importance: L1_ratio = 0, Lambda = 0.0464')
# plt.xticks(rotation=45)
# plt.savefig('data/Plots/Logistic_Feat_Imp.png')
#
# print('Tuned Cost Parameter: ' + str(pipe[2].C_))
# print('Tuned L1 ratio: ' + str(pipe[2].l1_ratio_))
#
# print('Training Accuracy: ' + str(pipe.score(X_train,y_train)))
#
# pipe.set_params(elasticnet__Cs=[pipe[2].C_])
# pipe.set_params(elasticnet__l1_ratios=pipe[2].l1_ratio_)
#
# y_pred = pipe.predict(X_test)
# print('Test Accuracy: ' + str(pipe.score(X_test,y_test)))
# print(classification_report(y_test, y_pred))
#
# plot_roc_curve(pipe,X_test,y_test)
# plt.title('Logistic Elastic Net: L1_Ratio = 0, Lambda = 0.0464')
# plt.savefig('data/Plots/Logistic_ROC_Curve.png')

# Support Vector Machine
# param_grid = {'C': [0.1, 1, 10, 100, 500, 1000],
#               'gamma': [0.18, 0.27, 0.67],
#               'kernel': ['linear','rbf']}
#
# svc = GridSearchCV(SVC(random_state=42), param_grid, refit = True, verbose = 0)
#
# pipe = Pipeline( [  ('impute',  knn_impute),
#                     ('scale',   scaler),
#                     ('SVM',     svc ) ])
#
#
# pipe.fit(X_train,y_train)
# print('Tuned Hyarameters: ' + str(pipe[2].best_params_))
# print('Training Accuracy: ' + str(pipe.score(X_train,y_train)))
#
# y_pred = pipe.predict(X_test)
# print('Test Accuracy: ' + str(pipe.score(X_test,y_test)))
# print(classification_report(y_test, y_pred))
#
# plot_roc_curve(pipe,X_test,y_test)
# plt.title('SVM w/ RBF Kernel, C = 10, Gamma = 0.67')
# plt.savefig('data/Plots/SVM_ROC_Curve.png')

# Penalized LDA
# lda = LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto')
# pipe = Pipeline( [  ('impute',  knn_impute),
#                     ('scale',   scaler),
#                     ('LDA',     lda ) ])
#
#
# pipe.fit(X_train,y_train)
# print('Training Accuracy: ' + str(pipe.score(X_train,y_train)))
#
# coefs = pd.DataFrame(pipe[2].coef_, columns=df_learn.columns)
# plt.bar(x=feat_names, height=abs(coefs.values.flatten())/(abs(coefs.values).sum()))
# plt.title('LDA Feature Importance')
# plt.xticks(rotation=45)
# plt.savefig('data/Plots/LDA_Feat_Imp.png')
#
# y_pred = pipe.predict(X_test)
# print('Test Accuracy: ' + str(pipe.score(X_test,y_test)))
# print(classification_report(y_test, y_pred))
#
# plot_roc_curve(pipe,X_test,y_test)
# plt.title('LDA ')
# plt.savefig('data/Plots/LDA_ROC_Curve.png')

# Random Forest
rf = RandomForestClassifier(n_estimators=500, criterion='gini',
                            max_features='sqrt', oob_score=True,
                            random_state=42)

pipe = Pipeline( [  ('impute',  knn_impute),
                    ('scale',   scaler),
                    ('rf',     rf ) ])


pipe.fit(X_train,y_train)
print('Training Accuracy: ' + str(pipe.score(X_train,y_train)))
print('OOB Error: ' + str(pipe[2].oob_score_))

# feat_imp = pipe[2].feature_importances_
# plt.bar(feat_names,feat_imp)
# plt.xticks(rotation=45)
# plt.title('Random Forest Classifier: Feature Importance (B = 500)')
# plt.savefig('data/Plots/RF_Feat_Imp.png')

y_pred = pipe.predict(X_test)
print('Test Accuracy: ' + str(pipe.score(X_test,y_test)))
print(classification_report(y_test, y_pred))

# plot_roc_curve(pipe,X_test,y_test)
# plt.title('Random Forest Classifier (B = 500)')
# plt.savefig('data/Plots/RF_ROC_Curve.png')


def Farm_Predictor(DNI, GHI, a40, a35, a30, Capacity, Longitude, Latitude):
    df = pd.DataFrame.from_dict({'DNI':[DNI], 'GHI':[GHI], 'a40':[a40],
                                 'a35':[a35], 'a30':[a30], 'Capacity':[Capacity],
                                 'Longitude':[Longitude], 'Latitude': [Latitude]})
    pred = pipe.predict_proba(df)[0]
    return {'Solar': pred[0], 'Wind': pred[1]}

Source = gr.inputs.Radio(['Solar','Wind'], label='Power Source')
DNI = gr.inputs.Slider(minimum=0, maximum=10, default=4, label="DNI")
GHI = gr.inputs.Slider(minimum=0, maximum=10, default=4, label="GHI")
a40 = gr.inputs.Slider(minimum=0, maximum=500, default=100, label="a40 Wind Potential (MW)")
a35 = gr.inputs.Slider(minimum=0, maximum=500, default=100, label="a35 Wind Potential (MW)")
a30 = gr.inputs.Slider(minimum=0, maximum=500, default=100, label="a30 Wind Potential (MW)")
Capacity = gr.inputs.Slider(minimum=0, maximum=2500, default=100, label="Capacity (MW)")
Longitude = gr.inputs.Slider(minimum=-172, maximum=145, default=100, label="Longitude")
Latitude = gr.inputs.Slider(minimum=13, maximum=50, default=20, label="Latitude")


gr.Interface(Farm_Predictor(), [DNI, GHI, a40, a35, a30, Capacity, Longitude, Latitude],
             "label", live=True).launch()


