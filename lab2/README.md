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