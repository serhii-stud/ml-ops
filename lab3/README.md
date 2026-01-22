### Feedback loop
```
IF data_drift_detected AND quality_ok:
    collect labels
ELIF quality_drop AND no_data_drift:
    concept_shift → revise labels / classes
ELIF both:
    retrain with feedback + manual labels
```
### Start airflow
```
docker-compose up -d airflow_etl
```

### Start mlflow
```
docker-compose up -d mlflow
```

### Train model
1. Ensure mlflow started
2. Start train script
```
docker-compose run --rm trainer python src/model/train.py
```

### Start prediction ms
Rebuild if any changes were made
```
docker-compose up -d --build prediction_service
```
or just start
```
docker-compose up -d prediction_service
```

### Start all
```
docker-compose up -d
```
THis starts only services. To train model you have to run script manually

TODO: add to etl model training 

### Stop services

Stop all
```
docker-compose down
```
Specific service
```
docker-compose stop <SERVICE_NAME>
```

### Review service logs
```
docker-compose logs <SERVICE_NAME>
```



### File structure
`/data` - initial train data

`/dags` - airflow dags

`/src/model` - code for model training

`src/microservices` - microcesrvices

`src/etl` - tasks for airflow dags

`ml_artifacs` - local db for mlflow(sqllight)

## Image for SM Endpoint
191072691166.dkr.ecr.us-east-1.amazonaws.com/mlflow-pyfunc:3.8.1
