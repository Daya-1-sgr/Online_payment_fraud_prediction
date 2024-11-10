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
            #feature_processor_path='artifacts/featureprocessor.pkl'
            preprocessor_path='artifacts/preprocessor.pkl'
            model=load_object(file_path=model_path)
           #feature_processor=load_object(file_path=feature_processor_path)
            preprocessor=load_object(file_path=preprocessor_path)
            
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e,sys)
# #'Unnamed: 0', 'step', 'type', 'amount', 'nameOrigFreq', 'oldbalanceOrg',
#        , 'nameDestFreq', 'oldbalanceDest', 
#        'isFraud', ],
#       dtype='object')

class CustomData:
    def __init__(self,step:int,type:str,amount:int,nameOrigFreq:int,oldbalanceOrg:int,
                 nameDestFreq:int,oldbalanceDest:int
                 ):
        self.step=step
        self.type=type
        self.amount=amount
        self.nameOrigFreq=nameOrigFreq
        self.oldbalanceOrg=oldbalanceOrg
        self.nameDestFreq=nameDestFreq
        self.oldbalanceDest=oldbalanceDest
        

    def get_data_as_frame(self):
        try:
            custom_input_data={'step':self.step,'type':self.type,'amount':self.amount,'nameOrigFreq':self.nameOrigFreq,
                               'oldbalanceOrg':self.oldbalanceOrg,'nameDestFreq':self.nameDestFreq,'oldbalanceDest':self.oldbalanceDest}
            return pd.DataFrame(custom_input_data,index=[0])
        except Exception as e:
            raise CustomException(e,sys)


