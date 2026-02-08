import logging
from IHML.utils import global_init as gi #Logging module initialized first
from IHML import globals as glb

from numpy import mean
from numpy import std

#Custom modules
import IHML.model_loader as ml
import IHML.data_loader as dl
import IHML.featureselector as fs
import IHML.base_model as bm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score,precision_score,roc_auc_score
from datetime import datetime
import timeit
import time
import pandas as pd

#Global loger
import logging
logger = logging.getLogger()

##This is PHD project for Onur Karadeli

def get_learner_names(M):
    str_learner_names=""
    for model in M:
        name = model.base_model.__class__.__name__
        str_learner_names+=name+","
    return str_learner_names


class IncrementalMetalearner(ClassifierMixin):

    logger=None
    neptune_run=None
    experiment_name= None
    M={} #Best Base learner list
    F={} #Best features list
    model=None #The trained model if any

    def __init__(self):
        self.logger = logging.getLogger()
        #self.set_up_neptune()

    def fit(self, X, Y,dataset=glb.DATASET):
        self._fit(X,Y,dataset)

    def predict(self,X):
        X=X[self.F]
        Y_pred = ml.predict_incremental_model(self.model, X )
        print("preds complete. Size:"+str(len(Y_pred)))
        return Y_pred
    def predict_proba(self,X):
        X = X[self.F]
        Y_pred = self.model.predict_proba(X)
        return Y_pred

    def _fit(self, X,Y,dataset):
        # get the models to evaluate
        models = ml.get_base_models()

        #Incremental MetaLearner
        M=list()
        #Incremental FeatureSet
        F=list()
        #Initial base model Accuracy
        H_base_models=list()
        #Top Features
        F_dict=dict()
        #Init variables
        score_current=0

        #get top alpha features
        for model in models:
            name = model.__class__.__name__
            model_top_features = fs.get_top_features(model, glb.ALPHA, X, Y,dataset)
            self.logger.info("Processing model:"+name)
            F_dict[model]=model_top_features

        # Evaluate the models and store initial results
        results, names = list(), list()
        for model in models:
            scores = ml.evaluate_model_train(model, X, Y)
            ##scores = ml.evaluate_model_train(model, X[F_dict[model]], Y) ##fixed a bug , ALL features was being input to model train , but it should be F instead.
            score=mean(scores)

            base_model_obj=bm.BaseModel(model,F_dict[model],glb.ALPHA)
            H_base_models.append((base_model_obj, score))

            results.append(scores)
            name=model.__class__.__name__
            names.append(name)
            #logger.info('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))

        #Sort models on accuracy
        ml.sort_models(H_base_models)

        #Incremental Learner building
        for model,score in H_base_models:
            #Empty Metalearner list (i.e. initialization) add first learner
            if not M:
                M.append(model)
                H_new_features = F_dict[model.base_model]
                F = fs.append_features([], H_new_features)
                self.logger.info("FIRST BEST Model:" + str(ml.get_model_name(model.base_model)) + " score:" + str(score))
                continue

            #Build new metalearner and new featureset
            M_new=list(M)
            M_new.append(model)
            F_new=list(F)
            H_new_features=F_dict[model.base_model]
            F_new=fs.append_features(F,H_new_features)

            #Get current and old metalearner scores
            previous_score=score_current
            score_current = ml.train_evaluate_incremental_model(M, F_new, X, Y)
            score_new =     ml.train_evaluate_incremental_model(M_new, F_new, X, Y)

            #Compare results to find out improvement
            score_improving= (score_new >= score_current)

            # Either Iteration validation score is improving Margins & Diversity is increasing
            if (score_improving):
                #There is improvement so update metalearner and featuresubset
                self.logger.info("IMPROVEMENT ! Adding new base learner.")
                M=M_new
                F=F_new
                self.logger.info("Score: "+str(score_new)+" Model count:"+str(len(M_new))+" feature count:"+str(len(F_new))+" new model:"+str(ml.get_model_name(model.base_model)))
            else:
                self.logger.info("NO IMPROVEMENT. Skipping, skipped model:"+str(ml.get_model_name(model.base_model)))

        #Best baselearner list
        self.logger.info("Best base learner list:"+str(len(M)))
        ml.print_model_list(M)
        self.logger.info("Best features list:"+str(len(F)))
        self.logger.info(F)
        self.logger.info("-----")
        self.M=M
        self.F=F
        self.model=ml.train_incremental_model(M,F,X,Y)


    def evaluate_experiment(self, X=None, Y=None):
        experiement_outputs = {}


        #Final Evaluation
        ##Single Models and StackedGeneralization
        models=ml.get_base_models()
        models.append(ml.get_stacked_model(models))
        #experiement_outputs["IncrementalStackedGeneraliztion"] = mean(scores)
        # Evaluate the models and store initial results
        results, names = list(), list()

        for model in models:
            experiement_start_time = time.time()##experiement_start_time = timeit.timeit()
            scores = ml.evaluate_model_train(model, X, Y)
            results.append(scores)
            name=model.__class__.__name__
            names.append(name)
            self.logger.info('>Test Score:%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
            #log results to NEPTUNE
            #self.neptune_run[name] = mean(scores)
            experiement_end_time = time.time()

            experiement_outputs[name]=str(round(mean(scores),4))+"/ TIME sec:"+str(round(experiement_end_time-experiement_start_time,4))
            experiement_outputs["DATASET_SIZE"] = len(X)

        ##Incremental Stack Generalization
        experiement_start_time = time.time()
        scores = ml.evaluate_model_train(self.model, X[self.F], Y)##BUG FIX here, previously it was X instead of X[self.F]
        experiement_end_time = time.time()
        self.logger.info('>Test Score:%s %.3f (%.3f)' % ("IncrementalStackedGeneraliztion", mean(scores), std(scores)))
        experiement_outputs["IncrementalStackedGeneraliztion"]=str(round(mean(scores),4))+"/ TIME sec:"+str(round(experiement_end_time-experiement_start_time,4))
        experiement_outputs["IHMLBestFeatureSet"] = str(self.F).replace("[","").replace("]","")
        experiement_outputs["IHMLBestLearnerSet"] = get_learner_names(self.M)



        #self.neptune_run["IncrementalStackedGeneraliztion"] = score
        #self.neptune_run.stop()
        self.logger.info("New Experiment is complete.")

        return  experiement_outputs

    def run_experiment(self,X=None,Y=None):

        self.logger.info("New Experiment is running...")
        glb.print_global_parameters()

        XTest, XTrain, YTest, YTrain = self.load_dataset()
        self.fit(XTrain,YTrain,glb.DATASET)
        experiment_outputs= self.evaluate_experiment(XTest,YTest)
        ##Calculate AUC/ROC...
        ##ml.evaluate_model_roc(self,XTest,YTest)

        return experiment_outputs

        #Now we built model, final test
        #preds=self.predict(XTest)
        #print("> Test score: IncrementalMetaLearner: {} %".format(100.0 * glb.SCORER(YTest,preds)))

    def load_dataset(self):
        # load dataset
        if glb.DATASET == "cancer":
            X, Y, target_column = dl.get_cancer_dataset()
        elif glb.DATASET == "iris":
            X, Y, target_column = dl.get_iris_dataset()
        elif glb.DATASET == "baddebt":
            X, Y, target_column = dl.get_baddebit_dataset()
        elif glb.DATASET == "magic":
            X, Y, target_column = dl.get_magic_dataset()
        elif glb.DATASET == "higgs":
            X, Y, target_column = dl.get_higgs_boson_dataset()
        elif glb.DATASET == "forest":
            X, Y, target_column = dl.get_forest_dataset()
        elif glb.DATASET == "sales":
            X, Y, target_column = dl.get_sales_dataset()
        elif glb.DATASET == "qsar":
            X, Y, target_column = dl.get_qsar_dataset()

        else:
            self.logger.info("INVALID DATASET, EXITING !!!")
            exit(-1)
        # self.logger.info("Calculating  p-values for dataset ")
        # original_df=pd.DataFrame(X)
        # original_df[target_column]=Y
        # self.calculate_p_values(target_column, original_df)
        # self.logger.info("Calculating  p-values for dataset COMPLETE")
        # Manual enabled code for SHAP values plotting
        # fs.plot_shap_values(glb.DecisionTreeClassifier(),X,Y)
        #

        logger.info("Sample size:"+str(len(X)))
        logger.info("Column size:" + str(len(X.columns)))

        XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.30, random_state=42)
        return XTrain, XTest, YTrain, YTest

    def set_params(self,params):
        #Changeable params
        if "DATASET" in params: glb.DATASET = params["DATASET"]
        if "ALPHA" in params: glb.ALPHA = params["ALPHA"]
        if "SCORER" in params:
            if params["SCORER"] == "accuracy":glb.SCORER=accuracy_score
            if params["SCORER"] == "f1": glb.SCORER = f1_score
            if params["SCORER"] == "recall":glb.SCORER=recall_score
            if params["SCORER"] == "precision": glb.SCORER = precision_score
            if params["SCORER"] == "roc_auc_score": glb.SCORER = roc_auc_score
        if "DATASET_SAMPLE_SIZE" in params: glb.DATASET_SAMPLE_SIZE = params["DATASET_SAMPLE_SIZE"]
        if "MODEL_LIST" in params and params["MODEL_LIST"] != None:
            glb.MODEL_LIST=glb.create_model_list(params["MODEL_LIST"])

        print("set experiment params complete")




    def calculate_p_values(self,target_column,data):
        import statsmodels.api as sm
        from sklearn.model_selection import train_test_split
        # If p-value < 0.05 -->Significant
        # If p-value > 0.05 -->Not Significant
        prices = data[target_column]
        features = data.drop(target_column, axis=1)
        X_train, X_test, Y_train, Y_test = train_test_split(features, prices, test_size=.2, random_state=10)
        x_incl_cons = sm.add_constant(X_train)
        model = sm.OLS(Y_train, x_incl_cons)  # ordinary least square
        results = model.fit()  # regresssion results
        # results.params
        # results.pvalues
        pvalues = pd.DataFrame({'coef': results.params, 'pvalue': round(results.pvalues, 3)})
        self.logger.info(pvalues)

    def calculate_model_scaleability(self):
        return 1






#logger = logging.getLogger()
#logger.info("*****")
#logger.info("DEV started.")
#logger.info("*****")

def experiment_model_scalability():
    global logger, model
    ##EXPERIMENT MODEL SCALABILITY
    logger = logging.getLogger()
    from xgboost import XGBClassifier
    scalability_test_results = {}
    for x in range(1, 20):
        glb.MODEL_LIST = set()
        ##add models to model list
        for y in range(x):
            model = XGBClassifier(verbosity=0, silent=True)
            glb.MODEL_LIST.add(model)
        # execute scalability experiment
        experiement_start_time = time.time()
        experiment.run_experiment()
        experiement_end_time = time.time()
        scalability_test_results[x] = int(experiement_end_time) - int(experiement_start_time)
        logger.info("SCALABILITY TEST COMPLETE number of models:" + str(
            len(glb.MODEL_LIST)) + " for dataset" + glb.DATASET + " is:" + str(
            int(experiement_end_time - experiement_start_time)))
    logger.info("SCALABILITY TEST RESULTS")
    logger.info(scalability_test_results)




if __name__ == '__main__':
    experiment = IncrementalMetalearner()
    experiment.run_experiment()
    #experiment_model_scalability() Test model scalability i.e. 1..20 models time cost





