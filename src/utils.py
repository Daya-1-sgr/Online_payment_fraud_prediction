import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import dill
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,matthews_corrcoef

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
        logging.info('Save object initiated')
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(x_train,y_train,x_test,y_test,models):
    try:
        report=[]
        for model_name,model in models.items():
            model.fit(x_train,y_train)
            y_train_pred=model.predict(x_train)
            y_test_pred=model.predict(x_test)

            tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
            precision = precision_score(y_test, y_test_pred)
            recall = recall_score(y_test, y_test_pred)
            f1 = f1_score(y_test, y_test_pred)
            mcc = matthews_corrcoef(y_test, y_test_pred)

            report.append({
                'Model': model_name,
                'TP': tp,
                'TN': tn,
                'FP': fp,
                'FN': fn,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'MCC': mcc
            })
        report_df=pd.DataFrame(report)    
        return report_df

    except Exception as e:
        raise CustomException(e,sys)