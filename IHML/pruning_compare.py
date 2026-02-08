#SKLearn sometimes throws warnings due to n_jobs not being supported in the future for KMeans. Just ignore them for now
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from PyPruning.RandomPruningClassifier import RandomPruningClassifier
from PyPruning.GreedyPruningClassifier import GreedyPruningClassifier
from incremental_metalearner import IncrementalMetalearner
import data_loader
import globals as glb
import logging
from xgboost import XGBClassifier
from PyPruning import Papers as papers


logger=logging.getLogger()

#data, target = load_digits(return_X_y = True)
#data, target = data_loader.get_baddebit_dataset()
data, target = data_loader.get_magic_dataset()
test_split=0.2
prune_split=0.3

XTP, Xtest, ytp, ytest = train_test_split(data, target, test_size=test_split, random_state=42)
Xtrain, Xprune, ytrain, yprune = train_test_split(XTP, ytp, test_size=prune_split, random_state=42)

print("size XTrain:"+str(len(Xtrain)))
print("size XPruneTrain:"+str(len(Xprune)))
print("size XTest:"+str(len(Xtest)))

n_base = 128
n_prune = 128
model = RandomForestClassifier(n_estimators=n_base)
model.fit(XTP, ytp)
pred = model.predict(Xtest)
logger.info("Accuracy of RF trained on XTrain + XPrune with {} estimators: {} %".format(n_base, 100.0 * accuracy_score(ytest, pred)))

model = RandomForestClassifier(n_estimators=n_base)
model.fit(Xtrain, ytrain)
pred = model.predict(Xtest)
logger.info("Accuracy of RF trained on XTrain only with {} estimators: {} %".format(n_base, 100.0 * accuracy_score(ytest, pred)))

########
##Test custom metric
pruned_model = papers.create_pruner("complementariness")
pruned_model.prune(Xtrain, ytrain, model.estimators_)
pred = pruned_model.predict(Xtest)
logger.info("Accuracy of complementariness with {} estimators: {} %".format(n_prune, 100.0 * accuracy_score(ytest, pred)))
######

########
##Test custom metric
pruned_model = papers.create_pruner("complementariness")
pruned_model.prune(Xtrain, ytrain, model.estimators_)
pred = pruned_model.predict(Xtest)
logger.info("Accuracy of complementariness with {} estimators: {} %".format(n_prune, 100.0 * accuracy_score(ytest, pred)))
######

########
##Test custom metric
pruned_model = papers.create_pruner("margin_distance")
pruned_model.prune(Xtrain, ytrain, model.estimators_)
pred = pruned_model.predict(Xtest)
logger.info("Accuracy of margin_distance with {} estimators: {} %".format(n_prune, 100.0 * accuracy_score(ytest, pred)))
######


pruned_model = RandomPruningClassifier(n_estimators = n_prune)
pruned_model.prune(Xprune, yprune, model.estimators_)
pred = pruned_model.predict(Xtest)
logger.info("Accuracy of RandomPruningClassifier with {} estimators: {} %".format(n_prune, 100.0 * accuracy_score(ytest, pred)))

pruned_model = GreedyPruningClassifier(n_prune)##, single_metric = "error")
pruned_model.prune(Xtrain, ytrain, model.estimators_)
pred = pruned_model.predict(Xtest)
logger.info("GreedyPruningClassifier with {} estimators and {} metric is {} %".format(n_prune, "model.estimators_", 100.0 * accuracy_score(ytest, pred)))

print("INC META LEARNER FINAL TEST !!!!")
incremental_ml = IncrementalMetalearner()
incremental_ml.fit(Xtrain, ytrain,"custom_dataset")
pred=incremental_ml.predict(Xtest)
logger.info("Accuracy of IncrementalMetaLearner: {} %".format(100.0 * accuracy_score(ytest,pred)))


#pruned_model = MIQPPruningClassifier(n_prune)##, single_metric = "error")
#pruned_model.prune(Xtrain, ytrain, model.estimators_)
#pred = pruned_model.predict(Xtest)
#print("MIQPPruningClassifier with {} estimators and {} metric is {} %".format(n_prune, model.estimators_, 100.0 * accuracy_score(ytest, pred)))