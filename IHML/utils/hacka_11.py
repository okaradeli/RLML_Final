from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, make_union
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from pandas import set_option
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer as Imputer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report,accuracy_score, f1_score
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn import model_selection
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.linear_model import Lasso, ElasticNet
#from fancyimpute import KNN
from sklearn.preprocessing import OneHotEncoder
from mlens.ensemble import SuperLearner


DO_SAMPLE=False
DO_TUNING=False
USE_SUPERLEARNER = False
DO_CROSSVALIDATION= False
TARGET_COLUMN = 'charged_off'

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

def calculate_model_accuracy(X_train, y_train, models):
    # Test options and evaluation metric
    num_folds = 3
    #scoring = 'accuracy'
    scoring  = "f1"

    results = []
    names = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=num_folds, random_state=42)
        cv_results = cross_val_score(model, X_train, y_train, cv=3, scoring=scoring)
        results.append(cv_results)
        print("kfold step complete name="+str(name)+" cv_results:"+str(cv_results))
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    return names, results


def get_base_models():
    basedModels = []
    #basedModels.append(('LR', LogisticRegression(penalty = 'l2', C = 0.01, random_state = 42)))
    #basedModels.append(('LDA', LinearDiscriminantAnalysis()))
    #basedModels.append(('KNN', KNeighborsClassifier()))
    #basedModels.append(('CART', DecisionTreeClassifier()))
    #basedModels.append(('NB', GaussianNB()))
    #basedModels.append(('SVM', SVC(probability=True)))
    #basedModels.append(('AB', AdaBoostClassifier()))
    #basedModels.append(('GBM', GradientBoostingClassifier()))
    basedModels.append(('XGB', XGBClassifier(seed=42,learning_rate=0.2)))
    #basedModels.append(('RF', RandomForestClassifier(random_state=42)))
    #basedModels.append(('ET', ExtraTreesClassifier()))

    return basedModels

def get_models():
    """Generate a library of base learners."""
    model1 = LogisticRegression(penalty = 'l2', C = 0.01, random_state = 42)
    param = {'n_neighbors': 15}
    model2 = KNeighborsClassifier(**param)
    model3 = SVC(**param)
    param = {'criterion': 'gini', 'max_depth': 3, 'max_features': 2, 'min_samples_leaf': 3}
    model4 = DecisionTreeClassifier(**param)
    param = {'learning_rate': 0.05, 'n_estimators': 150}
    model5 = AdaBoostClassifier(**param)
    param = {'learning_rate': 0.01, 'n_estimators': 100}
    model6 = GradientBoostingClassifier(**param)
    model7 = GaussianNB()
    model8 = RandomForestClassifier()
    model9 = ExtraTreesClassifier()
    model10 = XGBClassifier(random_state=42,learning_rate=0.2)

    models = { 'LR':model1,
              #'KNN':model2,
              #'SVC':model3,
              #'DT':model4,
              #'ADa':model5,
              #'GB':model6,
              #'NB':model7,
              'RF':model8,
              #'ET':model9,
              'XGB': model10
              }

    return models


def plotLearningCurves(estimator, X_train, X_test, y_train, y_test):
   X_values = [] # m values
   y_values_train = [] # accuracy_scores for train data
   y_values_test = [] # accuracy_scores for test data

   for i in range(20000, 200000, 20000):
       print("Iteration:"+str(i))
       dataToLearn = X_train[:i]
       dataToLearn_y = y_train[:i]
       estimator.fit(dataToLearn, dataToLearn_y)

       y_pred = estimator.predict(X_test)
       y_values_test.append(f1_score(y_test, y_pred))

       y_pred = estimator.predict(dataToLearn)
       y_values_train.append(f1_score(dataToLearn_y, y_pred))

       X_values.append(i)

   plt.plot(X_values, y_values_train, "r-", label="train f1 score")
   plt.plot(X_values, y_values_test, "b-", label="test f1 score")
#    plt.fig_size=(50,50)
   plt.legend()
   plt.show()


def draw_cm( actual, predicted ):
   cm = confusion_matrix( actual, predicted, [1,-1] )
   sns.heatmap(cm, annot=True,  fmt='.0f', xticklabels = ["1", "-1"] ,
               yticklabels = ["1", "-1"] )
   plt.ylabel('ACTUAL')
   plt.xlabel('PREDICTED')
   plt.show()



print("data loading...")

Y= pd.read_csv('./data/hacka/TrainPortion_Label_.csv')
X= pd.read_csv('./data/hacka/TrainPortion_Data.csv')
X.drop(TARGET_COLUMN, axis=1, inplace=True)##Somehow the X data contains the target columns so dropping
df_real_test= pd.read_csv('./data/hacka/TestPortion_Data.csv')


df = pd.concat([Y, X], axis=1, sort=False)
print("Raw data len:"+str(len(df)))
if DO_SAMPLE:
    df = df.sample ( 1000,random_state=42)
    print("Sampled Raw data len:" + str(len(df)))

#Engineer Term
df["term"] = (df.term.str.replace('months','').str.replace('s',' '))
df_real_test["term"] = (df_real_test.term.str.replace('months','').str.replace('s',' '))

def toInteger(row):
   rowstr= row['term']
   t_int = (int)(rowstr)
   return t_int
df["term_num"] = df.apply(toInteger, axis=1)
df_real_test["term_num"] = df_real_test.apply(toInteger, axis=1)

#Enginneer emp_length
df["emp_length"]=df.emp_length.str.replace('< 1 year','0.5').str.replace('years',' ').str.replace('year',' ').str.replace('+','')
df_real_test["emp_length"]=df_real_test.emp_length.str.replace('< 1 year','0.5').str.replace('years',' ').str.replace('year',' ').str.replace('+','')

def toFloat(row):
   rowstr= row['emp_length']
   t_int = (float)(rowstr)
   return t_int
df["emp_length_num"] = df.apply(toFloat, axis=1)
df_real_test["emp_length_num"] = df_real_test.apply(toFloat, axis=1)

#Enginneer emp_title
def isPresident(row):
  rowstr= (str)(row['emp_title'])
  president=0
  if "President" in rowstr or "president" in rowstr:
      president=1
  return president

def isDirector(row):
  rowstr= (str)(row['emp_title'])
  direct=0
  if "Director" in rowstr or "director" in rowstr:
      direct=1
  return direct

def isDriver(row):
  rowstr= (str)(row['emp_title'])
  driver=0
  if "Driver" in rowstr or "driver" in rowstr:
      driver=1
  return driver

df["isPresident"] = df.apply(isPresident, axis=1)
df["isDirector"] = df.apply(isDirector, axis=1)
df["isDriver"] = df.apply(isDriver, axis=1)

df_real_test["isPresident"] = df_real_test.apply(isPresident, axis=1)
df_real_test["isDirector"] = df_real_test.apply(isDirector, axis=1)
df_real_test["isDriver"] = df_real_test.apply(isDriver, axis=1)

def hasHistory(row):
  rowstr= (str)(row['total_bal_il'])
  if rowstr is None or rowstr=="":
      return 1
  return 0

df["has_history"] = df.apply(hasHistory, axis=1)
df_real_test["has_history"] = df_real_test.apply(hasHistory, axis=1)

def annualIncBoth(row):
  rowstr= (str)(row['application_type'])
  if rowstr =="Individual":
      return row['annual_inc']
  else:
      return row['annual_inc_joint']

df["annual_inc_both"] = df.apply(annualIncBoth, axis=1)
df_real_test["annual_inc_both"] = df_real_test.apply(annualIncBoth, axis=1)

def isEmployed(row):
  rowstr= (str)(row['emp_title'])
  if rowstr =="" or rowstr is None:
      return 0
  else :
      return 1

df["isEmployed"] = df.apply(isEmployed, axis=1)
df_real_test["isEmployed"] = df_real_test.apply(isEmployed, axis=1)

##Engineer earliest_cr_line
import datetime as dt
now = dt.datetime.now()
d = {"Jan":1, "Feb":2, "Mar":3, "Apr":4, "May":5, "Jun":6, "Jul":7, "Aug":8, "Sep":9, "Oct":10, "Nov":11, "Dec":12}
def calc(row):
   rowstr= row['earliest_cr_line']
   d_m = d[rowstr[:3]]
   y_m = (int)(rowstr[4:8])
   return ((now.year - y_m) * 12) + (now.month - d_m)
df["tenure"] = df.apply(calc, axis=1)
df_real_test["tenure"] = df_real_test.apply(calc, axis=1)

#DROP Unwanted columns
#df.drop(['inq_fi'],axis=1,inplace=True)
#df.drop(['inq_last_12m'],axis=1,inplace=True)
df.drop(['inq_last_6mths'],axis=1,inplace=True)
df_real_test.drop(['inq_last_6mths'],axis=1,inplace=True)
#df.drop(['total_bal_il'],axis=1,inplace=True)
#df.drop(['annual_inc_joint'],axis=1,inplace=True)
#df.drop(['dti_joint'],axis=1,inplace=True)
df.drop(['tot_cur_bal'],axis=1,inplace=True)#dropped due to correlation duplex
df_real_test.drop(['tot_cur_bal'],axis=1,inplace=True)

##SELECT FEATURES
Y = df[TARGET_COLUMN]

feature_columns = ['loan_amnt', 'installment', 'annual_inc', 'dti', 'delinq_2yrs', 'open_acc', 'pub_rec', 'total_acc', 'acc_now_delinq', 'tot_coll_amt', 'il_util', 'max_bal_bc', 'all_util', 'total_cu_tl', 'acc_open_past_24mths', 'avg_cur_bal', 'chargeoff_within_12_mths', 'delinq_amnt', 'mort_acc', 'pub_rec_bankruptcies', 'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit', 'term_num', 'emp_length_num', 'isPresident', 'isDirector', 'isDriver',
                   'has_history','annual_inc_both','isEmployed']
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df_num = df.select_dtypes(include=numerics)
df_real_test_num = df_real_test.select_dtypes(include=numerics)
X = df_num[feature_columns]
X_real = df_real_test_num[feature_columns]
category_columns  =['verification_status','home_ownership','purpose','addr_state','application_type']


### CLASIFICATION
#################
TARGET_COLUMN = 'charged_off'
Y = df[TARGET_COLUMN]

exclude_columns = ['charged_off','initial_list_status','earliest_cr_line','id','purpose','term','title','zip_code','addr_state','application_type','term','emp_title','emp_length','home_ownership','verification_status','desc','home_ownership','verification_status'                ]
exclude_columns_real = ['initial_list_status','earliest_cr_line','id','purpose','term','title','zip_code','addr_state','application_type','term','emp_title','emp_length','home_ownership','verification_status','desc','home_ownership','verification_status'                   ]

#One-hot encoder

encoder = OneHotEncoder(handle_unknown='ignore')
X_one_hot = encoder.fit_transform(df[category_columns])
encoder_df = pd.DataFrame(encoder.fit_transform(df[category_columns]).toarray())
df = df.join(encoder_df)
X = df.drop(exclude_columns_real, axis=1)
#Persist Data-engineered columns to excel and exit
X = X.dropna(axis=1)
X.to_csv('./data/hacka/Train_Data_Engineered_full_nona_3.csv')
print("Data Engineered data is persisted. Exitting...")
exit(0)

X_real_one_hot = encoder.transform(df_real_test[category_columns])
encoder_df_test = pd.DataFrame(encoder.fit_transform(df_real_test[category_columns]).toarray())
df_real_test = df_real_test.join(encoder_df_test)
X_real = df_real_test.drop(exclude_columns_real, axis=1)


X_train, X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
#X_train_feats = X_train [feature_columns]
#X_test_feats = X_test [feature_columns]
print("data processing complete.")

imputer = Imputer(strategy="median")
#print("Fancy imputer for KNN imputation")
#imputer = KNN(k=2)
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
X_real_test_imputed = imputer.transform(X_real)
#print("Fancy imputation complete.")

#Undersampling the minority class
#UNDERSAMPLING_RATIO= {1: 50, -1: 50}
#rus = RandomUnderSampler(random_state=0,ratio = 1.0)
rus = RandomUnderSampler(random_state=0)
##TODO fix RandomUnderSampler
X_resampled, y_resampled = rus.fit_resample(X_train_imputed, y_train)
#X_resampled, y_resampled = X_train_imputed, y_train

#scaling
sc = StandardScaler()
X_resampled = sc.fit_transform(X_resampled)
X_test_imputed = sc.transform(X_test_imputed)

X_real_test_imputed = sc.transform(X_real_test_imputed)

print("models starting...")
models=get_base_models()


#names,results = calculate_model_accuracy(X_train_imputed, y_train,models)
#names,results = calculate_model_accuracy(X_resampled, y_resampled,models)
#print(names)
#print(results)

def tune_xgb(X,Y):
    clf = XGBClassifier()
    # A parameter grid for XGBoost
    params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 2],
        'subsample': [0.6, 1.0],
        #'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 5]
    }
    folds = 2
    param_comb = 4

    random_search = RandomizedSearchCV(clf, param_distributions=params, n_iter=param_comb, scoring='f1', cv=folds, verbose=3, random_state=42)

    # Here we go
    random_search.fit(X, Y)

def tune_rf(X,Y):
    clf = RandomForestClassifier()
    params = {
        'min_child_weight': [1, 5, 10],
        'max_depth': [3, 5]
    }
    folds = 2
    param_comb = 4

    random_search = RandomizedSearchCV(clf, param_distributions=params, n_iter=param_comb, scoring='f1', cv=folds, verbose=3, random_state=42)

    # Here we go
    random_search.fit(X, Y)

#Tune XGB
if DO_TUNING:
    print("Tuning models...")
    tune_xgb(X_train_imputed,y_train)
    tune_rf(X_train_imputed,y_train)
    print("Tuning complete.")


#Single model accuracy
print("Calculating test f1 score")
#clf = RandomForestClassifier()
#clf = XGBClassifier(seed=42,subsample=0.6, min_child_weight=1, max_depth=5, gamma=2)
#clf = RandomForestClassifier(random_state = 42)#0.422
#clf = XGBClassifier(seed=42,learning_rate=0.2)#0.4761671892529414
#clf = ExtraTreesClassifier() ##0.41
#clf = KNeighborsClassifier(n_neighbors=3)##0.3678889259447789
#clf = SVM ( kernel = rbf ) #0.460
#clf = GradientBoostingClassifier()##0.4709808473

if not USE_SUPERLEARNER:
    clf = XGBClassifier(seed=42, learning_rate=0.2)
    #clf = LogisticRegression(penalty='l2', C=0.01, random_state=42)
    clf.fit(X_resampled, y_resampled)
    prediction = clf.predict(X_test_imputed)
    prediction_real = clf.predict(X_real_test_imputed)
    #probs = clf.predict_proba(X_test_imputed)
    #print("Tuning recall/prection")
    #prediction=[]
    #for p in probs[:, 1]:
    #   if p>0.6:
    #       prediction.append(1.)
    #   else:
    #       prediction.append(-1.)

if USE_SUPERLEARNER:
    # Ensembling ( Stacking )
    # Super learner
    base_learners = get_models()
    meta_learner = GradientBoostingClassifier(
        n_estimators=1000,
        loss="exponential",
        #max_features=6,
        #max_depth=3,
        subsample=0.5,
        learning_rate=0.001,
        random_state=42
    )

    # Instantiate the ensemble with 10 folds
    clf = SuperLearner(
        folds=3,
        random_state=42,
        verbose=2
        # ,backend="multiprocessing"
    )

    # Add the base learners and the meta learner
    clf.add(list(base_learners.values()), proba=True)
    clf.add_meta(meta_learner, proba=False)
    clf.fit(X_resampled,y_resampled)
    prediction = clf.predict(X_test_imputed)
    ###Predict the test set
    p_sl = clf.predict_proba(X_test_imputed)
    prediction=[]
    for p in p_sl[:, 1]:
       if p>0.5:
           prediction.append(1.)
       else:
           prediction.append(-1.)

    clf.fit(X_resampled,y_resampled)
    prediction = clf.predict(X_test_imputed)
    prediction_real = clf.predict(X_real_test_imputed)
    #SuperLearner 0.454

pd.DataFrame(prediction_real).to_csv("../backup/Group8_pred2.csv", index=False, header=False)

#plotLearningCurves(clf,X_resampled,X_test_imputed,y_resampled,y_test)
#models = get_base_models()
#names,results = calculate_model_accuracy(X_resampled, y_resampled,models)
#print(names)
#print(results)

print("plotting...")
#draw_cm( y_test, prediction )
print("plotting complete...")
print ('Accuracy:', accuracy_score(y_test, prediction))
print ('F1 score:', f1_score(y_test, prediction))
print ('Recall:', recall_score(y_test, prediction))
print ('Precision:', precision_score(y_test, prediction))
print ('\n clasification report:\n', classification_report(y_test,prediction))
print ('\n confussion matrix:\n',confusion_matrix(y_test, prediction))

#exit(0)

print("FEATURE IMPORTANCES")
features=X_train.columns
importances = clf.feature_importances_
indices = np.argsort(importances)

feature_importances = pd.DataFrame(clf.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances = feature_importances.head(100)



print(feature_importances)
plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
#plt.show()

print("plotting...")
#draw_cm( y_test, prediction )
print("plotting complete...")
print ('Accuracy:', accuracy_score(y_test, prediction))
print ('F1 score:', f1_score(y_test, prediction))
print ('Recall:', recall_score(y_test, prediction))
print ('Precision:', precision_score(y_test, prediction))
print ('\n clasification report:\n', classification_report(y_test,prediction))
print ('\n confussion matrix:\n',confusion_matrix(y_test, prediction))

df.head()