import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer
import re
import logging
from IHML import globals as glb
import numpy as np

logger = logging.getLogger()

# get the dataset
def get_baddebit_dataset():
    TARGET_COLUMN = "charged_off"

    logger.info("Baddebt data loading...")
    X = pd.read_csv('data/hacka/Train_Data_Engineered_full_nona_3.csv')
    #X = pd.read_csv('data/hacka/Train_Data_nona_undersampled_4.csv')
    logger.info("Raw data len:" + str(len(X)))
    X = X.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))##This stupid line is required for LGBM classifier
    if glb.DATASET_DO_SAMPLE:
        X = X.sample(glb.DATASET_SAMPLE_SIZE, random_state=42)
        logger.info("Sampled  data len:" + str(len(X))+" column size:"+str(len(X.columns)))
    ##convert Y column, -1 to 0
    X[TARGET_COLUMN]=np.where(X[TARGET_COLUMN] == -1, 0, X[TARGET_COLUMN])

    Y = X[TARGET_COLUMN]
    X.drop(TARGET_COLUMN, axis=1, inplace=True)

    if glb.EXPERIMENT_FEATURES:
        X=X[glb.EXPERIMENTAL_FEATURE_LIST]

    return X,Y,TARGET_COLUMN

# get the dataset
def get_sales_dataset():
    TARGET_COLUMN = "buy"

    logger.info("Sales data loading...")
    X = pd.read_csv('data/inbalanced/sales_data.csv')
    if glb.DATASET_DO_SAMPLE:
        X = X.sample(glb.DATASET_SAMPLE_SIZE, random_state=42)
        logger.info("Sampled  data len:" + str(len(X))+" column size:"+str(len(X.columns)))

    Y = X[TARGET_COLUMN]
    X.drop(TARGET_COLUMN, axis=1, inplace=True)

    if glb.EXPERIMENT_FEATURES:
        X=X[glb.EXPERIMENTAL_FEATURE_LIST]

    return X,Y,TARGET_COLUMN

# get the dataset
def get_higgs_boson_dataset():
    TARGET_COLUMN = "Label"
    ID_COLUMN = "EventId"
    TARGET_COLUMN2 = "Weight"


    logger.info("Higgs Boson data loading...")
    X = pd.read_csv('data/higgs/training.csv')
    logger.info("Raw data len:" + str(len(X)))
    X = X.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))##This stupid line is required for LGBM classifier
    if glb.DATASET_DO_SAMPLE:
        X = X.sample(glb.DATASET_SAMPLE_SIZE, random_state=42)
        logger.info("Sampled  data len:" + str(len(X))+" column size:"+str(len(X.columns)))

    mapping = {'b': 0, 's': 1}
    X = X.replace({'Label': mapping})

    X[TARGET_COLUMN]

    Y = X[TARGET_COLUMN]
    X.drop(TARGET_COLUMN, axis=1, inplace=True)

    X[ID_COLUMN] = 0
    X[TARGET_COLUMN2] = 0

    if glb.EXPERIMENT_FEATURES:
        logger.info("TESTING REDUCED FEATURE SET !!!!!!!!!!!!!!!!!!!!!!!!!")
        logger.info("Reduced Set: "+str(glb.EXPERIMENTAL_FEATURE_LIST))
        X=X[glb.EXPERIMENTAL_FEATURE_LIST]

    return X,Y,TARGET_COLUMN

# get the dataset
def get_cancer_dataset():
    #X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
    cancerdata = load_breast_cancer(as_frame=True)
    X=cancerdata.data
    Y=cancerdata.target
    logger.info("Cancer Data loaded X sample size:"+str(len(X))+" column size:"+str(len(X.columns)))
    return X, Y,cancerdata.target

def get_iris_dataset():
    data = load_iris(as_frame=True)
    X=data.data
    Y=data.target
    logger.info("Iris Data loaded X sample size:"+str(len(X))+" column size:"+str(len(X.columns)))
    return X, Y,data.target


def get_magic_dataset():
    #magic_path = download("http://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data", "magic.csv", tmpdir)
    logger.info("Magic data loading...")
    X = pd.read_csv('data/magic.csv')
    logger.info("Raw data len:" + str(len(X)))
    if glb.DATASET_DO_SAMPLE:
        X = X.sample(glb.DATASET_SAMPLE_SIZE)

    TARGET_COLUMN = "target"
    original_df = X

    X = X.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))##This stupid line is required for LGBM classifier
    Y = X[TARGET_COLUMN]
    X.drop(TARGET_COLUMN, axis=1, inplace=True)
    Y = np.array([0 if y == 'g' else 1 for y in Y])
    #Y = pd.Series(np.array([0 if y == 'g' else 1 for y in Y]))

    if glb.EXPERIMENT_FEATURES:
        X=X[glb.EXPERIMENTAL_FEATURE_LIST]

    return X,Y,TARGET_COLUMN

def get_forest_dataset():
    TARGET_COLUMN = "Cover_Type"

    logger.info("Forest Cover Type data loading...")
    X = pd.read_csv('data/forest/covtype.csv')
    logger.info("Raw data len:" + str(len(X)))
    X = X.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))##This stupid line is required for LGBM classifier
    if glb.DATASET_DO_SAMPLE:
        X = X.sample(glb.DATASET_SAMPLE_SIZE, random_state=42)
        logger.info("Sampled  data len:" + str(len(X))+" column size:"+str(len(X.columns)))



    Y = X[TARGET_COLUMN]
    Y = Y.astype(int) - 1  ##this stupid line is for the error: Expected: [0 1 2 3 4 5 6], got [1 2 3 4 5 6 7]

    X.drop(TARGET_COLUMN, axis=1, inplace=True)

    if glb.EXPERIMENT_FEATURES:
        X=X[glb.EXPERIMENTAL_FEATURE_LIST]

    return X,Y,TARGET_COLUMN


def get_qsar_dataset():
    TARGET_COLUMN = "result"

    logger.info("QSAR data loading...")
    #X = pd.read_excel('data/qsar_oral_toxicity.xlsx')
    X = pd.read_csv('data/qsar/qsar_oral_toxicity_csv.csv',sep=";")
    #X = pd.read_csv('data/qsar/qsar_oral_toxicity_large.csv', sep=";")
    #X = pd.read_csv('data/qsar/qsar_oral_toxicity_50k.csv', sep=";")

    logger.info("Raw data len:" + str(len(X)))
    if glb.DATASET_DO_SAMPLE:
        X = X.sample(glb.DATASET_SAMPLE_SIZE, random_state=42)
        logger.info("Sampled  data len:" + str(len(X))+" column size:"+str(len(X.columns)))

    # Feature sampling (TARGET her zaman korunur)
    if glb.DATASET_FEATURE_SIZE and glb.DATASET_FEATURE_SIZE < X.shape[1]:
        selected_features = (
            X.drop(columns=[TARGET_COLUMN])
            .sample(n=glb.DATASET_FEATURE_SIZE, axis=1, random_state=42)
            .columns.tolist()
        )
        X = X[selected_features + [TARGET_COLUMN]]
        logger.info("Sampled features (with target): " + str(len(X.columns)))

    mapping = {'negative': 0, 'positive': 1}
    X = X.replace({'result': mapping})

    Y = X[TARGET_COLUMN]
    X.drop(TARGET_COLUMN, axis=1, inplace=True)


    if glb.EXPERIMENT_FEATURES:
        X=X[glb.EXPERIMENTAL_FEATURE_LIST]

    return X,Y,TARGET_COLUMN



##test
if __name__ == '__main__':
    X,Y=get_higgs_boson_dataset()
    print("X size:"+str(len(X)))
    print("Y size:" + str(len(Y)))