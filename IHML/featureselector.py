# select top by shap value
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
import pickle
import logging
import IHML.globals as glb

logger = logging.getLogger()


def load_shap_features(model,dataset):
    model_name = model.__class__.__name__
    open_file = open(glb.SHAP_LIST_FOLDER+"_"+dataset+"_"+str(model_name[0:10])+".pkl", "rb")
    #list = pickle.load(open_file)
    list=pd.read_pickle(open_file)
    open_file.close()
    return list

def save_shap_features(model,list,dataset):
    model_name = model.__class__.__name__
    open_file = open(glb.SHAP_LIST_FOLDER+"_"+dataset+"_"+str(model_name[0:10])+".pkl", "wb")
    pickle.dump(list, open_file)
    open_file.close()
    return


def get_top_features(model,alpha, X, Y,dataset):

    if glb.LOAD_SHAP_FEATURES:
        try:
            features = load_shap_features(model,dataset)
            features = features.head(alpha)
            features_df = features.to_frame()
            logger.info("model:" + str(model)+" presaved features successfully loaded.")

            #Ugly code here ! Remove items from python series
            if glb.EXPERIMENT_FEATURES:
                for feature in features:
                    if feature not in glb.EXPERIMENTAL_FEATURE_LIST:
                        features_df = features_df[features_df.col_name != feature]
                features = features_df.squeeze()

            return features
        except:
            logger.info("Cant load pre-saved shap list. Recreating...")

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.5)
    model.fit(X_train, Y_train)

    #get N samples, since it takes too much time
    sample_shap_X_train=X_train.sample(n=glb.SHAP_SAMPLE_SIZE, random_state=1)
    explainer = shap.KernelExplainer(model.predict_proba, sample_shap_X_train)
    shap_values = explainer.shap_values(X_test, nsamples=glb.SHAP_SAMPLE_SIZE)

    vals = np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame(list(zip(X_test.columns, sum(vals))),columns=['col_name', 'feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    logger.info("feature importance:")
    logger.info(feature_importance)

    #save top features for later quick access
    save_shap_features(model,feature_importance['col_name'],dataset)

    #reduce features count to alpha
    feature_importance = feature_importance.head(alpha)

    return feature_importance['col_name']

def plot_shap_values(model,X, Y):

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.5)
    model.fit(X_train, Y_train)
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test)

    return

def get_top_features_mock(model, ALPHA, X, Y):
    features=list(X.columns)
    random.shuffle(features)
    features=features[0:ALPHA]
    return features

# Merge two feature sets,
# i.e. S1 = {a,b}
#      S2 = {b,c,d}
#      S1 U S2 = {a,b,c,d}
def append_features(F, F_new):
    set1=set(F)
    set2=set(F_new)
    set_union=set1.union(set2)
    return list(set_union)

def save_top_features(model):
    return True