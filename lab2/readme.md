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