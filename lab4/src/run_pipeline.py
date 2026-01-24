# pipelines/run_pipeline.py
import argparse
import sagemaker


# ... импорты ваших функций запуска (из ноутбука)

def run_data_prep(run_id):
    # Код из ячейки "ETL"
    pass


def run_training(run_id, data_version):
    # Код из ячейки "Train"
    pass


def run_evaluation(run_id, data_version):
    # Код из ячейки "Evaluation"
    # Добавить логику: если accuracy < threshold, вызывать exit(1) для остановки пайплайна
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, required=True)
    args = parser.parse_args()

    run_id = generate_run_id()
    run_data_prep(run_id)
    run_training(run_id)
    run_evaluation(run_id)
    # Если eval прошел успешно -> Deploy (или Register Model)