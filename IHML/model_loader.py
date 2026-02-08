import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
import IHML.globals as glb
import logging
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score, accuracy_score, recall_score,precision_score,roc_auc_score

logger = logging.getLogger()

# get a stacking ensemble of models
def get_base_models():
    all_models=list(glb.MODEL_LIST)
    #all_models.append(get_stacking(MODEL_LIST))
    return all_models



# get a list of models to eThe valuate
def get_stacking(models,base_model_extended=False):
    # define the base models
    level0 = list()
    # add base model
    i=0
    for model in models:
        i=i+1 #add some index to classifier name to avoid conflicts when there are 2 instances of i.e. RandomForest(40), RandomForest(60)
        if base_model_extended:
            level0.append((str(model.base_model.__class__.__name__)+str(i), model.get_base_learner()))
        else:
            level0.append((str(model.__class__.__name__)+str(i), model))

    # define meta learner model
    level1 = LogisticRegression(verbose=0)
    # define the stacking ensemble
    stacked_model = StackingClassifier(estimators=level0, final_estimator=level1, cv=2)
    return stacked_model

def get_stacked_model(models):
    base_models=list(models)
    stacked_model=get_stacking(base_models,False)
    return stacked_model

def get_extended_stacked_model(models):
    base_models=list(models)
    stacked_model=get_stacking(base_models,True)
    return stacked_model

def sort_models(H_base_scores):
    H_base_scores.sort(key=lambda x: x[1],reverse=True)


# evaluate a give model using cross-validation
def evaluate_model_train(model, X, Y):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)

    scorer=None
    if glb.SCORER==accuracy_score:
        scorer = make_scorer(glb.SCORER)
    else:
        scorer = make_scorer(glb.SCORER,labels=1,average='binary')

    #for multiclass(i.e. forest) and "precision" objective this error happens
    #ValueError: Target is multiclass but average='binary'. Please choose another average setting, one of [None, 'micro', 'macro', 'weighted'].
    if glb.DATASET=="forest" and (glb.SCORER in [f1_score,recall_score,precision_score]):##this dataset is MULTI-CLASS giving issues of Y values like: ValueError: Invalid classes inferred from unique values of y.  Expected: [0 1 2 3 4 5 6], got [1 2 3 4 5 6 7]  any idea ?
        scorer = make_scorer(glb.SCORER, average='macro')
        Y = Y-1

    if glb.DATASET=="forest" and (glb.SCORER in [roc_auc_score]):##this dataset is MULTI-CLASS giving issues of Y values like: ValueError: Invalid classes inferred from unique values of y.  Expected: [0 1 2 3 4 5 6], got [1 2 3 4 5 6 7]  any idea ?
        scorer = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovo')
        Y = Y - 1

    if glb.DATASET !="forest" and glb.SCORER in [roc_auc_score]:  ##this dataset is MULTI-CLASS giving issues of Y values like: ValueError: Invalid classes inferred from unique values of y.  Expected: [0 1 2 3 4 5 6], got [1 2 3 4 5 6 7]  any idea ?
        scorer = make_scorer(roc_auc_score, needs_proba=True)

    scores = cross_val_score(model, X, Y, scoring=scorer, cv=cv, n_jobs=-1, error_score='raise')
    logger.info("Scores for Variance Check:"+str(scores))
    return scores

# evaluate a give model using cross-validation
def evaluate_model_test(model,XTest,YTest):
    preds = model.predict(XTest)
    score = glb.SCORER(YTest, preds)
    return score

def evaluate_model_roc(model,x_test,y_test):
    import matplotlib.pyplot as plt
    from sklearn import metrics
    #y_pred=model.predict(x_test)
    y_pred = model.predict_proba(x_test)[:, 1]


    auc = metrics.roc_auc_score(y_test, y_pred)

    false_positive_rate, true_positive_rate, thresolds = metrics.roc_curve(y_test, y_pred)

    plt.figure(figsize=(10, 8), dpi=100)
    plt.axis('scaled')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title("AUC & ROC Curve")
    plt.plot(false_positive_rate, true_positive_rate, 'g')
    plt.fill_between(false_positive_rate, true_positive_rate, facecolor='lightgreen', alpha=0.7)
    plt.text(0.95, 0.05, 'AUC = %0.4f' % auc, ha='right', fontsize=12, weight='bold', color='blue')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()


def f1(actual, predicted, label):

    """ A helper function to calculate f1-score for the given `label` """
    # F1 = 2 * (precision * recall) / (precision + recall)
    tp = np.sum((actual==label) & (predicted==label))
    fp = np.sum((actual!=label) & (predicted==label))
    fn = np.sum((predicted!=label) & (actual==label))

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

# create a base learner and evaluate its score
def train_evaluate_incremental_model(meta_learner_list, feature_list, X, Y):
    X = X[feature_list]
    #stacked_model=get_stacked_model(meta_learner_list)
    stacked_model = get_extended_stacked_model(meta_learner_list)
    scores=evaluate_model_train(stacked_model, X, Y)
    return np.mean(scores)


# create a base learner and evaluate its score
def train_incremental_model(meta_learner_list, feature_list, X,Y):
    X = X[feature_list]
    stacked_model = get_extended_stacked_model(meta_learner_list)
    stacked_model.fit(X,Y)
    return stacked_model

# create a base learner and evaluate its score
def predict_incremental_model(stacked_model, X):
    return stacked_model.predict(X)

#
def train_predict_incremental_model(meta_learner_list, feature_list, X,Y):
    X = X[feature_list]
    stacked_model = get_extended_stacked_model(meta_learner_list)
    stacked_model.fit(X,Y)
    Y_pred=stacked_model.predict(X)
    return Y_pred

# create a base learner and evaluate its score
def train_stacked_model(meta_learner_list,feature_list,X,Y):
    X = X[feature_list]
    stacked_model=get_stacked_model(meta_learner_list)
    scores=evaluate_model_train(stacked_model, X, Y)
    return np.mean(scores)



def print_model_list(models):
    for model in models:
        logger.info(model.base_model.__class__.__name__)


def get_model_name(model):
    return model.__class__.__name__