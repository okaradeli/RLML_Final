from __future__ import annotations

import IHML.globals as glb
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar
import numpy as np
import random
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
import IHML.globals as glb
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from agent_RLML import dqn

#Global loger
import logging
logger = logging.getLogger()


ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")

class IHML_Env():
    in_training=False
    dataset=glb.DATASET
    max_epoch=100
    max_iter=50
    scorer=glb.accuracy_score


    # define meta learner model, TODO static single XGB
    level0 = list() #[]
    model=RandomForestClassifier()#model=XGBClassifier(random_state=13)
    level0.append((str(model.__class__.__name__) , model))
    level1 = LogisticRegression(verbose=0)
    # define the stacking ensemble
    stacked_model = StackingClassifier(estimators=level0, final_estimator=level1, cv=2)


    #Train Dataset references
    X=None
    Y=None

    #A reference to Agent object for logging purposes only
    agent=None


    # Features and Models enabled in action space
    action_vector ={}

    # All possible/available/visited actions
    action_space_all=[]
    action_space_remaining=[]
    action_space_visited=[]
    action_space_current_features_str = []
    action_space_current_bool = []
    epoch_current = 0 #used for curriculum learning calculations

    def __init__(self,X,Y,scorer=glb.SCORER,max_epoch=100,max_iter=50,):
        self.in_training = False
        self.dataset=glb.DATASET
        self.max_epoch=max_epoch
        self.max_iter=self.max_iter
        self.scorer=scorer
        self.X=X
        self.Y=Y
        self.epoch_current=0

        #these 2 are used for curriculum learning only
        self.X_original=X
        self.Y_original=Y



        # All possible/available/visited actions
        self.action_space_all = self.create_action_space(X, Y)
        self.action_space_remaining=self.action_space_all.copy()
        self.action_space_visited = []
        self.action_space_current_features_str = []
        self.action_space_current_bool = []



    def curriculum_learning_data_setup(self):
        # Curriculum Learning speed-up strategy
        if self.epoch_current == 0:#first epoch, train with whole data
            k = 1.0
        elif self.epoch_current>=glb.MAX_EPOCH-2:#last 2 epoch , train with whole data
            k = 1.0
        #else if self.epoch_current >=1 and self.epoch_current <=4:#mid-epochs train with partial data
        else:
            k = 0.25


        if glb.CURRICULUM_LEARNING_ENABLED:
            import pandas as pd
            import numpy as np
            len_before = len(self.X_original)

            # Step 1: Convert Y to a Series (if 1D) or DataFrame (if 2D)
            if self.Y_original.ndim == 1:
                Y_df = pd.Series(self.Y_original, name="target")

            # Step 2: Concatenate X and Y → Z
            Z = pd.concat([self.X_original.reset_index(drop=True), Y_df.reset_index(drop=True)], axis=1)

            # Step 4: Sample Z
            Z_sampled = Z.sample(frac=k, random_state=42).reset_index(drop=True)

            # Step 5: Split back into X' and Y'
            self.X = Z_sampled[self.X_original.columns]
            self.Y = Z_sampled.drop(columns=self.X_original.columns).to_numpy().ravel()

            len_after = len(self.X)
            logger.info(
                "Epoch:" + str(self.epoch_current) + " Curriculum Learning kicked-in: Before len(samples):" + str(
                    len_before) + " After len(samples):" + str(len_after))




    """Represent the state space as means of list of 0,1s
    """
    def get_action_to_feature(self,feature_index):
        return self.action_space_all[feature_index]

    """Represent the state space as means of list of 0,1s
    """

    def get_actions_to_features(self,action_space_all):
        feature_list=[]
        index=0
        for item in action_space_all:
            if item==1.0:
                feature = self.get_action_to_feature(index)
                if isinstance(feature, str):##only add features (i.e. skip Base Learners)
                    feature_list.append(feature)


            index = index+1
        return feature_list
    def get_binary_state_representation(self,input_list):

        # Create the binary representation
        binary_representation = [1 if item in self.action_space_all else 0 for item in self.action_space_all]

        return binary_representation

    def step(self, action: ActType,action_new,step_counter) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:


        #IHML Observation can be the number of features added so far.
        self.action_space_current_bool=action
        action_space_current_features_str=self.get_actions_to_features(self.action_space_current_bool)
        action_space_current_models = self.get_baselearners_from_action_vector(self.action_space_current_bool)
        action_space_current_models_str = [type(obj).__name__ for obj in action_space_current_models]

        self.action_vector[self.get_action_to_feature(action_new)]=0.0 #remove from the set
        observation =self.action_space_current_bool

        # Reward = Calculate accuracy
        #reward = random.random()
        reward = self.calculate_score (self.action_space_current_bool)
        logger.info("Reward:" + str(reward) +" Step:"+str(step_counter)  + " Features:" + str(action_space_current_features_str) +" Models:" + str(action_space_current_models_str))

        # If all actions are disabled , episode is completed
        all_actions_enabled = all(item == 0.0 for item in observation)
        terminated = all_actions_enabled

        # Example validity of the action (e.g., if the action is not possible, this could be False)
        truncated = False

        # Example additional information
        info = {
            "episode_length": 10,
            "current_score": 15
        }

        return (observation, reward, terminated, truncated, info)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None, ) -> tuple[
        ObsType, dict[str, Any]]:  # type: ignore
        """Resets the environment to an initial internal state, returning an initial observation and info.

        This method generates a new starting state often with some randomness to ensure that the agent explores the
        state space and learns a generalised policy about the environment. This randomness can be controlled
        with the ``seed`` parameter otherwise if the environment already has a random number generator and
        :meth:`reset` is called with ``seed=None``, the RNG is not reset.

        Therefore, :meth:`reset` should (in the typical use case) be called with a seed right after initialization and then never again.

        For Custom environments, the first line of :meth:`reset` should be ``super().reset(seed=seed)`` which implements
        the seeding correctly.

        .. versionchanged:: v0.25

            The ``return_info`` parameter was removed and now info is expected to be returned.

        Args:
            seed (optional int): The seed that is used to initialize the environment's PRNG (`np_random`).
                If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
                a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
                However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be reset.
                If you pass an integer, the PRNG will be reset even if it already exists.
                Usually, you want to pass an integer *right after the environment has been initialized and then never again*.
                Please refer to the minimal example above to see this paradigm in action.
            options (optional dict): Additional information to specify how the environment is reset (optional,
                depending on the specific environment)

        Returns:
            observation (ObsType): Observation of the initial state. This will be an element of :attr:`observation_space`
                (typically a numpy array) and is analogous to the observation returned by :meth:`step`.
            info (dictionary):  This dictionary contains auxiliary information complementing ``observation``. It should be analogous to
                the ``info`` returned by :meth:`step`.
        """
        # RESET all possible/available/visited actions
        self.action_space_remaining=self.action_space_all.copy()
        self.action_space_visited = []

        # Example observation: A grid or state representation
        #observation = [0.0, 0.0, 0.0, 0.0]  # Simplified example; replace with appropriate observation for your context
        observation = [float(1.0) for _ in self.action_space_all]

        logger.info("RESETTING THE ENVIRONMENT")
        self.action_space_current_bool=[float(1.0) for _ in self.action_space_all]

        # Set all elements to 0.0
        if(len(self.action_space_current_bool) > 1 ):

            self.action_space_current_bool = [1.0] * len(self.action_space_current_bool)


        # Example additional information
        info = {
            "episode_length": 10,
            "current_score": 15
        }

        return (self.action_space_current_bool, info)

    def action_space(self):
        return self.action_space_all

    def create_action_space(self,X,Y):
        action_space_list = []

        #For each feature in X, there is an add action
        for feature in X:
            action_space_list.append(feature)
            logger.info("Add Feature:" + str(feature) + " added to action space")
            self.action_vector[feature] = 1.0


        #for each base learner in globals.models there is an add action too
        for model in glb.MODEL_LIST:
            action_space_list.append(model)
            logger.info("Add Model:" + str(model.__class__.__name__) + " added to action space")
            self.action_vector[model]=1.0




        return action_space_list

    def get_features_from_action_vector(self,action_space_current):
        X_subset = action_space_current[len(glb.MODEL_LIST) :]
        return X_subset

    def get_baselearners_from_action_vector(self,action_vector):
        base_learner_list=[]
        #Find out which models are enabled in action vector
        for index,item in enumerate(action_vector):
            model_switch=item
            if model_switch==1.0:
                model=self.action_space_all[index]
                #check if this action is really a Model action
                if not isinstance(model, str):
                    base_learner_list.append(model)

        return base_learner_list


    def calculate_score(self, action_space_current):

        #Find out which "features" are enabled
        features_added = self.get_actions_to_features(action_space_current)
        X_subset = self.X[features_added]

        # Find out which "base models" are enabled
        models_added = self.get_baselearners_from_action_vector(self.action_space_current_bool)
        self.stacked_model=self.create_stacked_model(models_added)

        #Calculate performance with the candidate features and models
        scores=self.evaluate_model_train(self.stacked_model,X_subset,self.Y)

        #score is the mean value of validation scores
        return np.mean(scores)

    def evaluate_model_train(self,model, X, Y):
        #Use only the features coming from NN
        cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=1)

        scorer=None
        if glb.SCORER==accuracy_score:
            scorer = make_scorer(glb.SCORER)
        else:
            if glb.SCORER==glb.f1_score: scorer="f1"
            if glb.SCORER == glb.recall_score: scorer = "recall"
            if glb.SCORER == glb.precision_score: scorer = "precision_score"

            #scorer = make_scorer(glb.SCORER,labels=1,average='binary')
            scorer = make_scorer(glb.SCORER, average='binary')

        #for multiclass(i.e. forest) and "precision" objective this error happens, only "binary" works for multi-class f1
        if glb.DATASET=="forest" and glb.SCORER != accuracy_score:
            scorer = make_scorer(glb.SCORER,average='weighted') #Required for multicalss

        #The Curriculum Learning adjustments, we are giving number of samples based on the epoch index we are in
        #The smaller the Epoch, the more data we are feeding.

        scorer_auc = make_scorer(roc_auc_score) #Required for multicalss
        scores_auc = cross_val_score(model, X, Y, scoring=scorer_auc, cv=cv, n_jobs=-1, error_score='raise')



        scores = cross_val_score(model, X, Y, scoring=scorer, cv=cv, n_jobs=-1, error_score='raise')
        return scores

    def create_stacked_model(self,models_added):
        # define meta learner model,
        level0 = list()  # []

        #add enabled models
        for model in models_added:
            level0.append((str(model.__class__.__name__), model))

        level1 = LogisticRegression(verbose=0)
        # define the stacking ensemble
        stacked_model = StackingClassifier(estimators=level0, final_estimator=level1, cv=2)
        return stacked_model

