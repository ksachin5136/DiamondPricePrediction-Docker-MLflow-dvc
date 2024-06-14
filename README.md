# DiamondPricePrediction-Docker-MLflow-dvc

Commands

    source C:/Users/DELL/anaconda3/Scripts/activate condaEnv3JNB
    conda create -p venv python==3.10 -y
    conda activate ./venv
    pip install -r ./requirements.txt
    python ./training_pipeline.py
    python ./training_pipeline.py
    python ./training_pipeline.py
    mlflow ui
    pip install dagshub
    python ./training_pipeline.py
    python ./training_pipeline.py

Below code is Enough to Register the model on Remote server

                # Initialize DagsHub
                dagshub.init(repo_owner='ksachin5136', repo_name='DiamondPricePrediction-Docker-MLflow-dvc', mlflow=True)

                # You no longer need this line since `dagshub.init` sets it up
                # mlflow.set_registry_uri("https://dagshub.com/ksachin5136/DiamondPricePrediction-Docker-MLflow-dvc.mlflow")

                tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

                print("tracking_url_type_store:", tracking_url_type_store)

                with mlflow.start_run():

                    predicted_qualities = model.predict(X_test)

                    (rmse, mae, r2) = self.eval_metrics(y_test, predicted_qualities)

                    mlflow.log_metric("rmse", rmse)
                    mlflow.log_metric("r2", r2)
                    mlflow.log_metric("mae", mae)

                    # this condition is for the dagshub
                    # Model registry does not work with file store
                    if tracking_url_type_store != "file":

                        # Register the model
                        # There are other ways to use the Model Registry, which depends on the use case,
                        # please refer to the doc for more information:
                        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                        mlflow.sklearn.log_model(
                            model, "model", registered_model_name="ml_model")
                    # it is for the local
                    else:
                        mlflow.sklearn.log_model(model, "model")
