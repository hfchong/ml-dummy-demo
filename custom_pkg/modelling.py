from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

class CustomRF(RandomForestClassifier):
    def predict(self, X):
        return self.predict_proba(X)[:,1]


def _load_pyfunc(model_path):
    return joblib.load(model_path)