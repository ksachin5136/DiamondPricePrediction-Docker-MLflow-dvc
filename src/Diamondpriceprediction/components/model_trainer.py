
import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge,Lasso,ElasticNet
from src.Diamondpriceprediction.logger import logging
from src.Diamondpriceprediction.exception import customexception
from src.Diamondpriceprediction.utils.utils import save_object
from src.Diamondpriceprediction.utils.utils import evaluate_model



@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'Elasticnet':ElasticNet()
        }
            
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
          

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise customexception(e,sys)

# O/p 
# DELL@SachinKPC MINGW64 /d/Mytech/IDE_workspace/sachinji84git/iNeuronGit/fsdsmendtoend (main)
# $ python src/DimondPricePrediction/pipelines/training_pipeline.py 
# {'LinearRegression': 0.9359744723951412, 'Lasso': 0.9362418784297312, 'Ridge': 0.9359740831718698, 'Elasticnet': 0.8536982347288312}

# ====================================================================================

# Best Model Found , Model Name : Lasso , R2 Score : 0.9362418784297312

# ====================================================================================