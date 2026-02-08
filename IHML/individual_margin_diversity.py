import logging
import utils.global_init as gi #Logging module initialized first
import globals as glb
from functools import partial
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from scipy import spatial
from joblib import Parallel,delayed
import model_loader as ml
import data_loader as dl
import featureselector as fs
import base_model as bm
from sklearn.model_selection import train_test_split
from PyPruning.RankPruningClassifier import RankPruningClassifier, individual_margin_diversity
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

def individual_margin_diversity(i, ensemble_proba, target):
    alpha = 0.2
    """
    Computes the individual diversity of the classifier wrt. to the ensemble and its contribution to the margin. alpha controls the trade-off between both values. The original paper uses :math:`\\alpha = 0.2` in all experiments and reports that it worked well. Thus, it is also the default value here. You can supply a different ``alpha`` via the ``metric_options`` parameter of :class:`~PyPruning.RankPruningClassifier`.
    Reference:
        Guo, H., Liu, H., Li, R., Wu, C., Guo, Y., & Xu, M. (2018). Margin & diversity based ordering ensemble pruning. Neurocomputing, 275, 237–246. https://doi.org/10.1016/j.neucom.2017.06.052
    """
    iproba = ensemble_proba[i, :, :]
    n = iproba.shape[0]

    predictions = iproba.argmax(axis=1)
    V = np.zeros(ensemble_proba.shape)
    idx = ensemble_proba.argmax(axis=2)
    V[np.arange(ensemble_proba.shape[0])[:, None], np.arange(ensemble_proba.shape[1]), idx] = 1
    V = V.sum(axis=0)
    MDM = 0

    for j in range(n):
        if (predictions[j] == target[j]):
            # special case for margin: prediction for label with majority of votes
            if (predictions[j] == np.argmax(V[j, :])):
                # calculate margin with second highest number of votes
                sortedArray = np.sort(np.copy(V[j, :]))
                # check whether 1. and 2. max vot counts are equal! (margin = 0)
                if (sortedArray[-2] == np.max(V[j, :])):
                    margin = (V[j, target[j]] - (sortedArray[-2] - 1)) / n
                else:
                    margin = (V[j, target[j]] - sortedArray[-2]) / n
            else:
                # usual case for margin: prediction not label with majority of votes
                margin = (V[j, target[j]] - np.max(V[j, :])) / n
            # somehow theres still a rare case for margin == 0
            if (margin == 0):
                margin = 0.01

            fm = np.log(abs(margin))
            fd = np.log(V[j, target[j]] / n)
            MDM = MDM + (alpha * fm) + ((1 - alpha) * fd)
    return - 1.0 * MDM

def calculate_margin_diversity(proba, target,n_prune, data=None):
    n_received = len(proba)
    self_metric=individual_margin_diversity
    self_metric_options= {"alpha":0.2}
    n_jobs=5
    n_estimators=n_prune


    single_scores = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(self_metric)(i, proba, target) for i in range(n_received)
    )
    single_scores = np.array(single_scores)

    estimator_order = np.argpartition(single_scores, n_estimators)[:n_estimators]

    return estimator_order,single_scores


def test_model(model, Xprune, yprune, Xtest, ytest, estimators, name):
    print("Testing {}".format(name), end="", flush=True)
    model.prune(Xprune, yprune, estimators)
    pred = model.predict(Xtest)
    acc = accuracy_score(ytest, pred)
    print(" with accuracy {}".format(acc))

# evaluate a give model using cross-validation
def evaluate_model(model, X, Y):
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=1)
    scores = cross_val_score(model, X, Y, scoring=make_scorer(glb.SCORER), cv=cv, n_jobs=-1, error_score='raise')
    return scores

print("Starting INDIVIDUAL MARGIN DIVERSITY of base learners !!!")

# load dataset
if glb.DATASET == "cancer":
    X, Y = dl.get_cancer_dataset()
elif glb.DATASET == "iris":
    X, Y = dl.get_iris_dataset()
elif glb.DATASET == "baddebt":
    X, Y = dl.get_baddebit_dataset()
elif glb.DATASET == "magic":
    X, Y = dl.get_magic_dataset()
else:
    print("INVALID DATASET, EXITING !!!")
    exit(-1)


test_split=0.2
prune_split=0.2


XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=test_split, random_state=42)
XTrain_original = pd.DataFrame(XTrain)
YTrain_original = np.copy(YTrain)

XTrain, XPrune, YTrain, YPrune = train_test_split(XTrain, YTrain, test_size=prune_split, random_state=42)
print("size XTrain:"+str(len(XTrain)))
print("size XPruneTrain:"+str(len(XPrune)))
print("size XTest:"+str(len(XTest)))

#first fit base learners
fitted_models=[]
for base_model in glb.MODEL_LIST:
    base_model.fit(XTrain,YTrain)
    fitted_models.append(base_model)
print("Models fitted succesfully.")

# Get MDM scores & probabas of each base_model
n_classes = 2
n_prune=5
n_prune_incremental=0

index_info = np.array([0,1])
proba = np.zeros(shape=(len(fitted_models), XTrain.shape[0], n_classes), dtype=np.float32)
for i, e in enumerate(fitted_models):
    proba[i, :, index_info] = e.predict_proba(XTrain).T
base_model_diversity_order, base_model_diversity =calculate_margin_diversity(proba,YTrain_original,n_prune_incremental)
base_models_list = list(glb.MODEL_LIST)
pruned_model_list = [base_models_list[index] for index in base_model_diversity_order]
glb.MDM_MODEL_LIST=pruned_model_list

print("Calculating accuracy of base models...")
for base_model in glb.MODEL_LIST:
    scores=evaluate_model(base_model,XTest,YTest)
    print("Scores_"+str(base_model)+":"+str(np.mean(scores)))

print("Training inc_metalearner")
from incremental_metalearner import IncrementalMetalearner
inc_model = IncrementalMetalearner()
inc_model.fit(XTrain_original,YTrain_original)
pred = inc_model.predict(XTest)
acc = accuracy_score(YTest, pred)
print("Testing inc_meta accuracy:"+str(acc))

#Rank based MD pruner
#pruned_model = RankPruningClassifier(metric=individual_margin_diversity, n_estimators=n_prune)
#print("RankPruningClassifier")
#test_model(pruned_model, XPrune, YPrune, XTest, YTest, fitted_models, "invidiual_margin_diversity")

#Vanilla Stacked Generalization ensemble
stacked_model = ml.get_stacked_model(glb.MODEL_LIST)
stacked_model.fit(XTrain_original,YTrain_original)
pred = stacked_model.predict(XTest)
acc = accuracy_score(YTest, pred)
print("Scores_StackedGenerilization"+ ":" + str(acc))

print("Prune compare complete  !!!")