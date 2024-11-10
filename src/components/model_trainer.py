import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models,save_report_to_csv

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import matthews_corrcoef
@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class Modeltrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info('splitting training and test input data')
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],train_arr[:,-1],
                test_arr[:,:-1],test_arr[:,-1]
            )
            models={
                
                'LogisticRegression':LogisticRegression(class_weight='balanced',max_iter=1000),
                'LGBMClassifier':lgb.LGBMClassifier(class_weight='balanced'),
                'XGBClassifier':xgb.XGBClassifier(scale_pos_weight=0.001),
                
                'CatBoostClassifier':CatBoostClassifier(class_weights=[1854,1774935]),

                'RandomForestClassifier':RandomForestClassifier(class_weight='balanced',n_jobs=-1),
                
                
              
                
            }
            model_report=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)
            print(model_report)
            
            model_report['MCC'] = pd.to_numeric(model_report['MCC'], errors='coerce')
            save_report_to_csv(model_report)
            model_report_sorted=model_report.sort_values(by='MCC',ascending=False)
            best_model_score=model_report_sorted['MCC'].iloc[0]
            best_model_name=model_report_sorted['model'].iloc[0]
            best_model=models[best_model_name]

            logging.info(f'Best model found is {best_model_name} with score {best_model_score}')

            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)
            predicted=best_model.predict(x_test)
            mcc=matthews_corrcoef(y_test,predicted)

            return mcc
        except Exception as e:
            raise CustomException(e,sys)

