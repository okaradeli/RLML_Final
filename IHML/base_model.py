import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
#from mlxtend.feature_selection import ColumnSelector
import IHML.model_loader as ml
import logging
import IHML.globals as glb

class BaseModel:
    #Actual Base Learning algorithm ( i.e. KNN, RF...)
    base_model = None

    #Default top N features
    alpha=50

    def __init__(self, model,top_features,alpha=50):
        self.base_model = model
        self.top_features = top_features
        self.alpha=alpha

    def get_base_learner(self):

        if glb.BASE_MODEL_EXTENDED:
            pipeline = Pipeline([
                ('fs', ColumnSelector(cols=self.top_features)),
                ('clf', self.base_model)])
        else:
            pipeline = Pipeline([
                ('clf', self.base_model)])


        return pipeline