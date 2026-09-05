"""Export the sklearn TF-IDF + LogisticRegression pipeline to ONNX (FP32),
then apply dynamic INT8 quantization to it.

Run from Milestone-2/backend/:
    python optimize/export_onnx.py
"""

import joblib
from onnxruntime.quantization import QuantType, quantize_dynamic
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from sklearn.pipeline import Pipeline

FP32_PATH = "optimize/model_fp32.onnx"
INT8_PATH = "optimize/model_int8.onnx"


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

    quantize_dynamic(FP32_PATH, INT8_PATH, weight_type=QuantType.QInt8)
    print(f"wrote {INT8_PATH}")


if __name__ == "__main__":
    main()
