import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


Sites = pd.read_csv('data/cleandata/MergedIncomplete.csv')
Solar = pd.read_csv('data/cleandata/SolarPotential.csv')
Wind  = pd.read_csv('data/cleandata/WindPotential.csv')

tol1 = 0.25
tol2 = 0.25

for val in range(Sites.shape[0]):
    success = 0
    coord   = 0
    while success == 0:
        if coord == Solar.shape[0]:
            Sites.loc[val,'GHI'] = np.nan
            Sites.loc[val,'DNI'] = np.nan
            success += 1
        elif Solar.iloc[coord,0]-tol1 <= Sites.iloc[val,2] <= Solar.iloc[coord,0]+tol1:
            if Solar.iloc[coord,1]-tol2 <= Sites.iloc[val,3] <= Solar.iloc[coord,1]+tol2:
                Sites.loc[val,'GHI'] = Solar.iloc[coord,2]
                Sites.loc[val,'DNI'] = Solar.iloc[coord,3]
                success += 1
            else:
                coord += 1
        else:
            coord += 1


tol1 = 0.4
tol2 = 0.4

for val in range(Sites.shape[0]):
    success = 0
    coord   = 0
    while success == 0:
        if coord == Wind.shape[0]:
            Sites.loc[val,'a30'] = np.nan
            Sites.loc[val,'a35'] = np.nan
            Sites.loc[val,'a40'] = np.nan
            success += 1
        elif Wind.iloc[coord,0]-tol1 <= Sites.iloc[val,2] <= Wind.iloc[coord,0]+tol1:
            if Wind.iloc[coord,1]-tol2 <= Sites.iloc[val,3] <= Wind.iloc[coord,1]+tol2:
                Sites.loc[val,'a30'] = Wind.iloc[coord,3]
                Sites.loc[val,'a35'] = Wind.iloc[coord,4]
                Sites.loc[val,'a40'] = Wind.iloc[coord,5]
                success += 1
            else:
                coord += 1
        else:
            coord += 1


Sites.to_csv('data/cleandata/MergedComplete.csv',index=False)


Sites[['State', 'Power Source']] = Sites[['State', 'Power Source']].astype('category')
df_learn = Sites.drop(['Power Source', 'State'] ,axis=1)

X_train, X_test, y_train, y_test = \
    train_test_split(df_learn, Sites['Power Source'],
                     test_size=0.3,
                     random_state=42)

Train = pd.concat([X_train, y_train], axis=1)
Test = pd.concat([X_test, y_test], axis=1)
Train.to_csv('data/cleandata/Train.csv',index=False)
Test.to_csv('data/cleandata/Test.csv',index=False)
