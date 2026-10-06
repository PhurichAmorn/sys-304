"""Export the sklearn TF-IDF + LogisticRegression pipeline to ONNX (FP32).

Run from Milestone-2/backend/:
    python optimize/export_onnx.py
"""

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from sklearn.pipeline import Pipeline

FP32_PATH = "../model/model_fp32.onnx"


def main():
    vectorizer = joblib.load("vectorizer.joblib")
    model = joblib.load("model.joblib")
    pipeline = Pipeline([("tfidf", vectorizer), ("clf", model)])

    onnx_model = convert_sklearn(
        pipeline, initial_types=[("text_input", StringTensorType([None, 1]))]
    )
    with open(FP32_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"wrote {FP32_PATH}")


if __name__ == "__main__":
    main()
