import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import dill
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,matthews_corrcoef
from datetime import datetime

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
            y_test_pred=model.predict(x_test)

            tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
            precision = precision_score(y_test, y_test_pred)
            recall = recall_score(y_test, y_test_pred)
            f1 = f1_score(y_test, y_test_pred)
            mcc = matthews_corrcoef(y_test, y_test_pred)
            print('saving metrics for the model:',model_name)
            report.append([model_name,tp,tn,fp,fn,precision,recall,f1,mcc])
            print('saved model:',model_name)
        report_df=pd.DataFrame(report,columns=['model','TP','TN','FP','FN','Precision','Recall','F1 Score','MCC'])    
        return report_df

    except Exception as e:
        raise CustomException(e,sys)
    
def save_report_to_csv(report_df, output_dir='outputs/evaluation_reports'):
    """
    Saves the evaluation report to a CSV file.
    Adds a timestamp to the filename.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    file_path = os.path.join(output_dir, f"evaluation_report_{timestamp}.csv")
    
    try:
        report_df.to_csv(file_path, index=False)
        print(f"Report saved successfully at {file_path}")
    except Exception as e:
        raise CustomException(e,sys)