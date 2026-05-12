import mlflow
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/sivakumar026/mlops_pipeline.mlflow")


dagshub.init(repo_owner='sivakumar026', repo_name='mlops_pipeline', mlflow=True)

with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)