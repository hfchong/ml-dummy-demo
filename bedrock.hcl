version = "1.0"

train {
    image = "asia.gcr.io/span-ai/pyspark:v2.4.0r2"
    install = ["pip3 install -r requirements.txt"]
    script = ["python3 demo.py"]

    parameters {
        STORAGE_PATH = "gs://ml-dummy-data"
        FEATURES_FILE = "features.csv"
        LABELS_FILE = "labels.csv"
        OUTPUT_MODEL_NAME = "custom_model.pkl"
        N_ESTIMATORS_LIST = "2, 5, 10"
        MAX_DEPTH_LIST = "4, 8, 12"
        MIN_SAMPLES_SPLIT_LIST = "2, 4"
    }
}

serve {
    image = "python:3.7"
    install = ["pip3 install -r requirements.txt"]
    script = ["python3 serve_http.py"]
}