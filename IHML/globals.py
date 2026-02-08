from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import logging
from sklearn.metrics import f1_score,recall_score,precision_score
from sklearn.metrics import accuracy_score

import torch
from torch import nn
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu', epochs=50, lr=0.01, batch_size=32):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self._build_model()

    def _build_model(self):
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }

        if self.activation not in activation_map:
            raise ValueError(f"Unsupported activation: {self.activation}")

        layers = []
        in_features = self.input_size

        for hidden in self.hidden_sizes:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(activation_map[self.activation])
            in_features = hidden

        layers.append(nn.Linear(in_features, self.output_size))
        self.model = nn.Sequential(*layers)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            for X_batch, y_batch in loader:
                y_pred = self.model(X_batch)
                loss = self.loss_fn(y_pred, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(X)
            return torch.argmax(logits, dim=1).numpy()

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))




#Dataset related (baddebt,cancer,iris,magic, ...)
DATASET="baddebt"
#DATASET="higgs"
#DATASET="magic"
#DATASET="cancer"
#DATASET="forest"
#DATASET="sales"
#DATASET="qsar"

DATASET_DO_SAMPLE=True
DATASET_SAMPLE_SIZE=30000 ##Default sample size , generally overriden by Experiment parameter
DATASET_FEATURE_SIZE=100  ##used for scalability tests only

#Experiments related
#EXPERIMENT_NUMBER = "113_higgs"
EXPERIMENT_NUMBER = "211_base"
EXPERIMENTS_INPUT_FILE="./experiments/experiments_input_IHML_v"+str(EXPERIMENT_NUMBER)+".xlsx"
EXPERIMENTS_OUTPUT_FILE="./experiments/experiments_output_IHML_v"+str(EXPERIMENT_NUMBER)+".xlsx"
EXPERIMENT_ID = 0 ##By default a single experiment is executed ( train and/or run) but there can be batch experiments where ID is dynamic coming from Excel
MODEL_DATA_FILE_NAME = "./model_data/model_" + DATASET + "_v"+str(EXPERIMENT_NUMBER)+".dat" ##Torch saved NN model after train

MAX_EPOCH=50 #Default number of max epochs during training
MAX_ITERATION=8 #Default max iterations during an episode (epoch)
CURRICULUM_LEARNING_ENABLED=True

#Model Building-feature selection related
ALPHA=10
#SCORER=f1_score
SCORER=accuracy_score


#Base model specific features (True) or all set of features (False)
BASE_MODEL_EXTENDED=False
#Improvement Type (Restricted or Any)
SCORE_IMPROVEMENT_TYPE="Any"

#Maturity Scorer Improvement Type (Restricted or Any)
MATURITY_SCORER="in"

logging.getLogger("lightgbm").setLevel(logging.WARNING)  # LightGBM logs are annoying so we disable them
#MODEL_LIST_STRING="XGBClassifier"
#MODEL_LIST_STRING="XGBClassifier" #by default RF is already added TODO
#MODEL_LIST_STRING="LGBMClassifier,KNeighborsClassifier,XGBClassifier"#by default RF is already added TODO
#MODEL_LIST_STRING="RandomForestClassifier,XGBClassifier,DecisionTreeClassifier,KNeighborsClassifier,GaussianNB,LGBMClassifier"
#MODEL_LIST_STRING="RandomForestClassifier,XGBClassifier,KNeighborsClassifier,GaussianNB"
#MODEL_LIST_STRING="RandomForestClassifier,XGBClassifier,KNeighborsClassifier,GaussianNB"
MODEL_LIST_STRING="XGBClassifier,RandomForestClassifier,DecisionTreeClassifier"
#MODEL_LIST_STRING="RandomForestClassifier"


def create_ANN_classifier():
    from keras.models import Sequential
    from keras.layers import Dense

    classifier = Sequential()
    # Defining the Input layer and FIRST hidden layer,both are same!
    # relu means Rectifier linear unit function
    classifier.add(Dense(units=10, input_dim=9, kernel_initializer='uniform', activation='relu'))

    # Defining the SECOND hidden layer, here we have not defined input because it is
    # second layer and it will get input as the output of first hidden layer
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

    # Defining the Output layer
    # sigmoid means sigmoid activation function
    # for Multiclass classification the activation ='softmax'
    # And output_dim will be equal to the number of factor levels
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

    # Optimizer== the algorithm of SGG to keep updating weights
    # loss== the loss function to measure the accuracy
    # metrics== the way we will compare the accuracy after each step of SGD
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return classifier

def build_mlp_classifier(input_size, hidden_sizes, output_size, **kwargs):
    return TorchMLPClassifier(input_size, hidden_sizes, output_size, **kwargs)

def create_model_list(models):
    base_models=[]
    for model in models.split(","):
        if(model=="RandomForestClassifier"):base_models.append(RandomForestClassifier())
        if(model == "DecisionTreeClassifier"): base_models.append(DecisionTreeClassifier())
        if (model == "KNeighborsClassifier"): base_models.append(KNeighborsClassifier())
        if (model == "GaussianNB"): base_models.append(GaussianNB())
        if (model == "LGBMClassifier"): base_models.append(LGBMClassifier())
        if (model == "XGBClassifier"): base_models.append(XGBClassifier(verbosity=0, silent=True))
        if (model== "ANNClassifier"): base_models.append(build_mlp_classifier(input_size=4, hidden_sizes=[64, 32], output_size=3, activation='relu'))
        ##if (model== "ANNClassifier"): base_models.append()

    return base_models

MODEL_LIST=create_model_list(MODEL_LIST_STRING)

EXPERIMENT_FEATURES=False #True if enable testing i.e. StackingClassifier with reduced (previously found) feature set
EXPERIMENTAL_FEATURE_LIST=['PRI_jet_leading_eta', 'DER_mass_transverse_met_lep', 'DER_prodeta_jet_jet', 'DER_mass_vis', 'DER_mass_MMC', 'PRI_tau_pt', 'DER_met_phi_centrality', 'PRI_jet_subleading_pt', 'PRI_met', 'DER_pt_h', 'PRI_tau_eta', 'EventId', 'DER_pt_ratio_lep_tau', 'PRI_jet_leading_phi', 'PRI_jet_leading_pt', 'PRI_jet_all_pt', 'Weight', 'PRI_jet_subleading_phi', 'DER_deltar_tau_lep', 'DER_sum_pt', 'PRI_lep_pt', 'PRI_lep_eta', 'DER_mass_jet_jet', 'PRI_met_sumet']

#Margin and Diversitiy ordered Ensemble sorted models
MDM_MODEL_LIST=[]

#Feature selection related
SHAP_SAMPLE_SIZE=50
LOAD_SHAP_FEATURES=True
SHAP_LIST_FOLDER="./resources/shap_list"

def print_global_parameters():
    logger = logging.getLogger()
    logger.info("ALPHA="+str(ALPHA))
    logger.info("SCORER=" + str(SCORER))
    logger.info("DATASET=" + str(DATASET))
    logger.info("DO_SAMPLE=" + str(DATASET_DO_SAMPLE))
    logger.info("SHAP_SAMPLE_SIZE=" + str(SHAP_SAMPLE_SIZE))
    logger.info("LOAD_SHAP_FEATURES=" + str(LOAD_SHAP_FEATURES))
    logger.info("SHAP_LIST_FOLDER=" + str(SHAP_LIST_FOLDER))

def get_global_parameters():
    params = {"DATASET": DATASET,
              "DATASET_DO_SAMPLE": DATASET_DO_SAMPLE,
              "SCORER": SCORER,
              "ALPHA": ALPHA,
              "SHAP_SAMPLE_SIZE": SHAP_SAMPLE_SIZE,
              "MODEL_LIST": MODEL_LIST,
              "BASE_MODEL_EXTENDED": BASE_MODEL_EXTENDED,
              "SCORE_IMPROVEMENT_TYPE": SCORE_IMPROVEMENT_TYPE
              }
    return params



