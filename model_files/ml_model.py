import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

##Put all the functions here
#preprocess the origin column and map them to the countries
def preprocess_origin_col(df):
    df["origin"] = df["origin"].map({1: "India", 2: "USA", 3: "Germnay"})
    return df


#Adding Attributes using BaseEstimator and Transformer
#acceleration on power and acceleration on cylinder
#cylinder_index= 0, horsepower_index=2, acceleration_index= 4

class CustomAttrAdder(BaseEstimator, TransformerMixin):
    def __init__(self, acc_on_power= True):
        self.acc_on_power = acc_on_power
    def fit(self, X, y=None):
        return self #Nothing to do here
    def transform(self, X):
        acc_on_cyl = X[:,4]/ X[:,0]
        if self.acc_on_power:
            acc_on_power = X[:,4]/ X[:,2]
            return np.c_[X, acc_on_power,acc_on_cyl]
        return np.c_[X, acc_on_cyl]



##pipeline for numerical attributes
##imputing -> adding attributes -> scale them
def num_pipeline_transformer(data):
    
    '''
    Function to process numerical transformations
    Argument:
        data: original dataframe 
    Returns:
        num_attrs: numerical dataframe
        num_pipeline: numerical pipeline object
        
    '''
    num_attr = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration','model year']
    
    num_pipeline = Pipeline([
        ("Impute", SimpleImputer(strategy="median")),
        ("Attr Added", CustomAttrAdder()),
        ("Scaling", StandardScaler()),
])
    return num_attr, num_pipeline


def full_pipeline_transformer(data):
    
    '''
    function to process the entire transformation for both numerical and categorical data
    Argument: 
        data: original dataframe 
    Returns:
        
    '''
    cat_attr = ["origin"]
    num_attr, num_pipeline = num_pipeline_transformer(data)
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attr),
        ("cat", OneHotEncoder(), cat_attr),
])
    prepared_data = full_pipeline.fit_transform(data)
    return prepared_data



#final function: to cover this entire flow
def predict_mpg(config, best_estimator):
    
    if type(config) == dict:
        df = pd.DataFrame(config)
    else:
        df = config
    
    preproc_df = preprocess_origin_col(df)
    prepared_df = full_pipeline_transformer(preproc_df)
    y_pred = best_estimator.predict(prepared_df)
    return y_pred