import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

df_train = pd.read_csv('data/cleandata/Train.csv')
df_test = pd.read_csv('data/cleandata/Test.csv')

X_train = df_train.drop(['Power Source'],axis=1)
y_train  = df_train['Power Source']
X_test = df_test.drop(['Power Source'],axis=1)
y_test  = df_test['Power Source']

feat_names = X_train.columns


imputer = IterativeImputer(sample_posterior=True,
                            imputation_order='descending',
                            skip_complete=True,
                            random_state=42)
scaler = StandardScaler()


# Logistic Elastic Net
log_elastic_net = LogisticRegressionCV(cv=5,
                                       penalty='elasticnet',
                                       solver='saga',
                                       random_state=42,
                                       max_iter=10000,
                                       l1_ratios=[0,0.2,0.4,0.6,0.8,.9,1] )

pipeEN = Pipeline( [('impute',  imputer),
                    ('scale',   scaler),
                    ('elasticnet', log_elastic_net ) ])

pipeEN.fit(X_train,y_train)
coefs = pd.DataFrame(pipeEN[2].coef_, columns=feat_names)
Logistic_feat_imp = abs(coefs.values.flatten())/(abs(coefs.values).sum())

tuned_C  = pipeEN[2].C_
tuned_L1 = pipeEN[2].l1_ratio_
print('Logistic EN Tuned Cost Parameter: ' + str(tuned_C))
print('Logistic EN Tuned L1 ratio: ' + str(tuned_L1))
print('Logistic EN Training Accuracy: ' + str(pipeEN.score(X_train,y_train)))

pipeEN.set_params(elasticnet__Cs=[pipeEN[2].C_])
pipeEN.set_params(elasticnet__l1_ratios=pipeEN[2].l1_ratio_)


y_pred = pipeEN.predict(X_test)
disp = plot_roc_curve(pipeEN, X_test, y_test,
                      name='Logistic EN (L1=' + str(float(tuned_L1)) + ' lambda=' + str(round(1/float(tuned_C),4)))
Logistic_acc = pipeEN.score(X_test,y_test)
print('Logistic EN Test Accuracy: ' + str(Logistic_acc))
print(classification_report(y_test, y_pred))

del pipeEN, y_pred, coefs, log_elastic_net

# Support Vector Machine
param_grid = {'C': [0.1, 1, 10, 100, 500, 1000],
              'gamma': [0.18, 0.27, 0.67],
              'kernel': ['linear','rbf']}

svc = GridSearchCV(SVC(random_state=42), param_grid, refit = True, verbose = 0)

pipeSVM = Pipeline( [  ('impute',  imputer),
                    ('scale',   scaler),
                    ('SVM',     svc ) ])


pipeSVM.fit(X_train,y_train)
SVM_acc = pipeSVM.score(X_train,y_train)
tuned_SVM = pipeSVM[2].best_params_
print('SVM Tuned Hyarameters: ' + str(tuned_SVM))
print('SVM Training Accuracy: ' + str(SVM_acc))

y_pred = pipeSVM.predict(X_test)
SVM_disp = plot_roc_curve(pipeSVM, X_test, y_test,
                          ax=disp.ax_,name='SVM (rbf, C=' + str(tuned_SVM.get('C'))
                                           + ' ,gamma=' + str(tuned_SVM.get('gamma')) + ')')
print('SVM Test Accuracy: ' + str(pipeSVM.score(X_test,y_test)))
print(classification_report(y_test, y_pred))

del pipeSVM, param_grid, y_pred, svc


# Penalized LDA
lda = LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto')
pipeLDA = Pipeline( [  ('impute',  imputer),
                    ('scale',   scaler),
                    ('LDA',     lda ) ])


pipeLDA.fit(X_train,y_train)
print('LDA Training Accuracy: ' + str(pipeLDA.score(X_train,y_train)))

coefs = pd.DataFrame(pipeLDA[2].coef_, columns=feat_names)
LDA_feat_imp=abs(coefs.values.flatten())/(abs(coefs.values).sum())

y_pred = pipeLDA.predict(X_test)
LDA_disp = plot_roc_curve(pipeLDA, X_test, y_test, ax=disp.ax_,name='LDA')
LDA_acc  = pipeLDA.score(X_test,y_test)
print('LDA Test Accuracy: ' + str(LDA_acc))
print(classification_report(y_test, y_pred))

del pipeLDA, y_pred, coefs, lda

# Random Forest
rf = RandomForestClassifier(n_estimators=500, criterion='gini',
                            max_features='sqrt', oob_score=True,
                            random_state=42)

pipeRF = Pipeline( [  ('impute',  imputer),
                      ('rf',     rf ) ])


pipeRF.fit(X_train,y_train)
print('RF Training Accuracy: ' + str(pipeRF.score(X_train,y_train)))
print('RF OOB Accuracy: ' + str(pipeRF[1].oob_score_))

importances = permutation_importance(pipeRF,X_train,y_train,
                                               n_repeats=20,random_state=42)

RF_feat_imp = importances.importances_mean/importances.importances_mean.sum()

y_pred = pipeRF.predict(X_test)
RF_disp = plot_roc_curve(pipeRF, X_test, y_test, ax=disp.ax_,name='Random Forest (B=500)')
RF_acc  = pipeRF.score(X_test,y_test)
print('RF Test Accuracy: ' + str(RF_acc))
print(classification_report(y_test, y_pred))

del pipeRF, y_pred, rf


RF_disp.figure_.suptitle("ROC Curve Comparison")
plt.tight_layout()
plt.show()
plt.savefig('data/Plots/ROC_Curves.png')

plt.figure()
plt.plot(feat_names, Logistic_feat_imp, 'bo', linestyle='None',
         label='Logistic EN (L1= ' + str(float(tuned_L1)) + ' Lambda= ' + str(round(1/float(tuned_C),4)) + ')')
plt.plot(feat_names, RF_feat_imp, 'r+', linestyle='None',label='Random Forest (B=500)')
plt.plot(feat_names, LDA_feat_imp, 'gv', linestyle='None',label='LDA')
plt.xticks(rotation=20)
plt.title('Feature Importance Comparison')
plt.legend(frameon=False)
plt.savefig('data/Plots/Feat_Imp.png')
plt.show()

plt.figure()
heights = [Logistic_acc,SVM_acc,LDA_acc,RF_acc]
models  = ['Logistic EN (' + str(round(Logistic_acc,4)) + ')' ,
           'SVM (' + str(round(SVM_acc,4)) + ')',
           'LDA (' + str(round(LDA_acc,4)) + ')',
           'Random Forest (' + str(round(RF_acc,4)) + ')']
plt.bar(x = models, height = heights, color='g')
plt.xticks(rotation=45)
plt.title('Accuracy Comparison')
plt.tight_layout()
plt.savefig('data/Plots/Accuracy.png')
plt.show()


