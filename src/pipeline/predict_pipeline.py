import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path='artifacts/model.pkl'
            feature_processor_path='artifacts/featureprocessor.pkl'
            preprocessor_path='artifacts/preprocessor.pkl'
            model=load_object(file_path=model_path)
            feature_processor=load_object(file_path=feature_processor_path)
            preprocessor=load_object(file_path=preprocessor_path)
            
        except Exception as e:
            raise CustomException(e,sys)
