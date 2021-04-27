import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

df = pd.read_csv('data/cleandata/MergedComplete.csv')

df[['State', 'Power Source']] = df[['State', 'Power Source']].astype('category')
df_learn = df.drop(['Power Source', 'State'] ,axis=1)

X_train, X_test, y_train, y_test = \
    train_test_split(df_learn, df['Power Source'],
                     test_size=0.3,
                     random_state=42)


knn_impute = KNNImputer()
scaler = StandardScaler()

# TODO: Penalized LDA, Trees/Bagging, Boosting, Plots (ROC curves, variable importance)

# Logistic Elastic Net
# log_elastic_net = LogisticRegressionCV(cv=5,
#                                        penalty='elasticnet',
#                                        solver='saga',
#                                        random_state=42,
#                                        max_iter=10000,
#                                        l1_ratios=[0.0001,0.1,0.12,0.14,0.16,0.18,0.2] )
#
# pipe = Pipeline( [  ('impute',  knn_impute),
#                     ('scale',   scaler),
#                     ('elasticnet', log_elastic_net ) ])
#
# pipe.fit(X_train,y_train)
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



# Support Vector Classifier/Machine
# param_grid = {'C': [0.1, 1, 10, 100, 1000],
#               'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
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


# Penalized LDA

# Trees/Bagging/Boosting