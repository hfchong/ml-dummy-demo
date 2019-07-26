"""
Script for serving.
"""
import joblib
import json
import os
import socketserver
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler

import numpy as np

from custom_pkg.modelling import CustomRF

SERVER_PORT = int(os.environ.get("SERVER_PORT", "8080"))
MODEL = joblib.load('custom_model.pkl')
FEATURE_NAMES = ['age', 'gender', 'x', 'z', 'y', 'c', 'b', 'a']


def predict_prob(features,
                 model=MODEL):
    """Predict probability given features.

    Args:
        features (dict)

        model (CustomRF)
    Returns:
        prob (float): probability
    """
    row_feats = np.asarray(
        [[features[feature_name] for feature_name in FEATURE_NAMES]]
    )

    # Score
    prob = model.predict_proba(row_feats)[:, 1].item()

    return prob


# pylint: disable=invalid-name
class Handler(SimpleHTTPRequestHandler):
    """Handler for http requests"""

    def do_POST(self):
        """Returns the `churn_prob` given the subscriber features"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        self.send_response(HTTPStatus.OK)
        self.end_headers()

        features = json.loads(post_data.decode("utf-8"))
        result = {
            "prediction_prob": predict_prob(features)
        }
        self.wfile.write(bytes(json.dumps(result).encode('utf-8')))


def main():
    """Starts the Http server"""
    print(f"Starting server at {SERVER_PORT}")
    httpd = socketserver.TCPServer(("", SERVER_PORT), Handler)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
