import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

TARGET_COLUMN = 'charged_off'

X = pd.read_csv('../data/hacka/Train_Data_Engineered_full_nona_3.csv')
print("data size:"+str(len(X))+" rows count. Column count:"+ str(len(X.columns)) )
Y = X[TARGET_COLUMN]
X.drop(TARGET_COLUMN, axis=1, inplace=True)

rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_resample(X, Y)

df=pd.DataFrame(X_resampled)
df[TARGET_COLUMN]=y_resampled
df = df.dropna()
#df.drop(['id'], axis=1, inplace=True)
df = df.sample(frac=1).reset_index(drop=True)

print("resampled & nona data size:"+str(len(df))+" rows count. Column count:"+ str(len(df.columns)) )
df.to_csv('../data/hacka/Train_Data_nona_undersampled_4.csv')

print("Resampled data is persisted. Exitting...")
