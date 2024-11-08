import sys
import os
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder,FunctionTransformer
from src.utils import save_object

@dataclass
class DataTransformerConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
    featureprocessor_obj_file_path=os.path.join('artifacts','featureprocessor.pkl')

class Data_Transformation:
    def __init__(self):
        self.data_transformation_config=DataTransformerConfig()
   
    def get_data_transformer_object(self):
        '''responsible for data transformation'''
        logging.info('Data transformer started')
        try:
            def drop_columns_func(X):
                drop_columns=['Unnamed: 0','newbalanceOrig','newbalanceDest']
                return X.drop(columns=drop_columns) 
            def add_feature_func(X):
                frequency_encoding_orig = X['nameOrig'].value_counts().to_dict()
                X['nameOrigFreq'] = X['nameOrig'].map(frequency_encoding_orig)
                frequency_encoding_dest = X['nameDest'].value_counts().to_dict()
                X['nameDestFreq'] = X['nameDest'].map(frequency_encoding_dest)
                return X
            featurepipeline=Pipeline(steps=[('feature_adder',FunctionTransformer(add_feature_func)),
                                            ('dropper',FunctionTransformer(drop_columns_func))])
            num_cols=[ 'step','amount','oldbalanceOrg','oldbalanceDest','nameOrigFreq','nameDestFreq']
            cat_cols=['type']
            num_pipeline=Pipeline(steps=[('imputer',SimpleImputer(strategy='mean')),
                                         ('scaler',StandardScaler())])
            cat_pipeline=Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')),
                                         ('onehot',OneHotEncoder(sparse_output=False)),
                                         ('scaler',StandardScaler(with_mean=False))])
            logging.info('Pipelines are complete')
            preprocessor = ColumnTransformer(transformers=[('num_pipeline', num_pipeline, num_cols),
                                                           ('cat_pipeline', cat_pipeline, cat_cols)])

    

            return (featurepipeline,preprocessor)
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info('Reading test data and train data completed')
            featurePipeline_obj,preprocessor_obj=self.get_data_transformer_object()
            target_column_name='isFraud'
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            print(input_feature_train_df.columns)
            logging.info('applying  feature pipeline and preprocessing object on training and testing dfs')
            input_feature_train_df_feature_pipe=featurePipeline_obj.fit_transform(input_feature_train_df)
            input_feature_test_df_feature_pipe=featurePipeline_obj.transform(input_feature_test_df)
            logging.info('feature pipeline complete')


            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df_feature_pipe)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df_feature_pipe)
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            logging.info('columntransformer complete')

            logging.info('saving preprocessing object and feature processor object')
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,obj=preprocessor_obj)
            save_object(file_path=self.data_transformation_config.featureprocessor_obj_file_path,obj=featurePipeline_obj)

            return(train_arr,test_arr)
        except Exception as e:
            raise CustomException(e,sys)