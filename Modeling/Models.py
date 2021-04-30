import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, plot_roc_curve, roc_curve,roc_auc_score
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
log_elastic_net = LogisticRegressionCV(cv=5,
                                       penalty='elasticnet',
                                       solver='saga',
                                       random_state=42,
                                       max_iter=10000,
                                       l1_ratios=[0,0.2,0.4,0.6,0.8,.9,1] )

pipe = Pipeline( [  ('impute',  knn_impute),
                    ('scale',   scaler),
                    ('elasticnet', log_elastic_net ) ])

pipe.fit(X_train,y_train)
coefs = pd.DataFrame(pipe[2].coef_, columns=df_learn.columns)
Logistic_feat_imp = abs(coefs.values.flatten())/(abs(coefs.values).sum())

print('Logistic EN Tuned Cost Parameter: ' + str(pipe[2].C_))
print('Logistic EN Tuned L1 ratio: ' + str(pipe[2].l1_ratio_))

print('Logistic EN Training Accuracy: ' + str(pipe.score(X_train,y_train)))

pipe.set_params(elasticnet__Cs=[pipe[2].C_])
pipe.set_params(elasticnet__l1_ratios=pipe[2].l1_ratio_)


y_pred = pipe.predict(X_test)
disp = plot_roc_curve(pipe, X_test, y_test, name='Logistic EN (L1=0,lambda=0.0464)')
Logistic_acc = pipe.score(X_test,y_test)
print('Logistic EN Test Accuracy: ' + str(Logistic_acc))
print(classification_report(y_test, y_pred))


# Support Vector Machine
param_grid = {'C': [0.1, 1, 10, 100, 500, 1000],
              'gamma': [0.18, 0.27, 0.67],
              'kernel': ['linear','rbf']}

svc = GridSearchCV(SVC(random_state=42), param_grid, refit = True, verbose = 0)

pipe = Pipeline( [  ('impute',  knn_impute),
                    ('scale',   scaler),
                    ('SVM',     svc ) ])


pipe.fit(X_train,y_train)
SVM_acc = pipe.score(X_train,y_train)
print('SVM Tuned Hyarameters: ' + str(pipe[2].best_params_))
print('SVM Training Accuracy: ' + str(SVM_acc))

y_pred = pipe.predict(X_test)
SVM_disp = plot_roc_curve(pipe, X_test, y_test, ax=disp.ax_,name='SVM (rbf, C=10, gamma=0.67)')
print('SVM Test Accuracy: ' + str(pipe.score(X_test,y_test)))
print(classification_report(y_test, y_pred))


# Penalized LDA
lda = LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto')
pipe = Pipeline( [  ('impute',  knn_impute),
                    ('scale',   scaler),
                    ('LDA',     lda ) ])


pipe.fit(X_train,y_train)
print('LDA Training Accuracy: ' + str(pipe.score(X_train,y_train)))

coefs = pd.DataFrame(pipe[2].coef_, columns=df_learn.columns)
LDA_feat_imp=abs(coefs.values.flatten())/(abs(coefs.values).sum())

y_pred = pipe.predict(X_test)
LDA_disp = plot_roc_curve(pipe, X_test, y_test, ax=disp.ax_,name='LDA')
LDA_acc  = pipe.score(X_test,y_test)
print('LDA Test Accuracy: ' + str(LDA_acc))
print(classification_report(y_test, y_pred))


# Random Forest
rf = RandomForestClassifier(n_estimators=500, criterion='gini',
                            max_features='sqrt', oob_score=True,
                            random_state=42)

pipe = Pipeline( [  ('impute',  knn_impute),
                    ('scale',   scaler),
                    ('rf',     rf ) ])


pipe.fit(X_train,y_train)
print('RF Training Accuracy: ' + str(pipe.score(X_train,y_train)))
print('RF OOB Accuracy: ' + str(pipe[2].oob_score_))

RF_feat_imp = pipe[2].feature_importances_

y_pred = pipe.predict(X_test)
RF_disp = plot_roc_curve(pipe, X_test, y_test, ax=disp.ax_,name='Random Forest (B=500)')
RF_acc  = pipe.score(X_test,y_test)
print('RF Test Accuracy: ' + str(RF_acc))
print(classification_report(y_test, y_pred))


RF_disp.figure_.suptitle("ROC curve comparison")
plt.show()
plt.savefig('data/Plots/ROC_Curves.png')

plt.figure()
plt.plot(feat_names, Logistic_feat_imp, 'bo', linestyle='None',
         label='Logistic EN (L1=0, Lambda=0.0464)')
plt.plot(feat_names, RF_feat_imp, 'r+', linestyle='None',label='Random Forest (B=500)')
plt.plot(feat_names, LDA_feat_imp, 'gv', linestyle='None',label='LDA')
plt.xticks(rotation=45)
plt.legend(frameon=False)
plt.title('Feature Importance Comparision')
plt.savefig('data/Plots/Feat_Imp.png')
plt.show()

plt.figure()
plt.bar(x = ['Logistic EN', 'SVM', 'LDA', 'Random Forest'],
        height = [Logistic_acc,SVM_acc,LDA_acc,RF_acc], color='g')
plt.xticks(rotation=45)
plt.title('Accuracy Comparision')
plt.savefig('data/Plots/Accuracy.png')
plt.show()

# gradio
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


gr.Interface(Farm_Predictor, [DNI, GHI, a40, a35, a30, Capacity, Longitude, Latitude],
             "label", live=True).launch()


