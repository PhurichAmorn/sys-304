"""Create a real INT8 classifier while retaining floating-point text preprocessing.

Run from backend/: python optimize/export_int8.py
"""

import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as rt
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic

SOURCE = Path(__file__).resolve().parents[2] / "model/model_fp32.onnx"
DESTINATION = Path(__file__).with_name("model_int8.onnx")


def main():
    graph = onnx.load(SOURCE)
    classifiers = [n for n in graph.graph.node if n.op_type == "LinearClassifier"]
    if len(classifiers) != 1:
        raise ValueError("expected one LinearClassifier")
    classifier = classifiers[0]
    attributes = {a.name: helper.get_attribute_value(a) for a in classifier.attribute}
    if (attributes["classlabels_ints"] != [0, 1]
            or attributes["post_transform"] != b"LOGISTIC"
            or attributes["multi_class"] != 0):
        raise ValueError("expected the existing binary logistic classifier")

    # ORT's dynamic quantizer handles MatMul, not ai.onnx.ml LinearClassifier.
    weights = np.asarray(attributes["coefficients"], dtype=np.float32).reshape(2, -1).T
    bias = np.asarray(attributes["intercepts"], dtype=np.float32)
    graph.graph.initializer.extend([
        numpy_helper.from_array(weights, "int8_classifier_weights"),
        numpy_helper.from_array(bias, "int8_classifier_bias"),
    ])
    graph.graph.value_info.extend([
        helper.make_tensor_value_info(classifier.input[0], TensorProto.FLOAT,
                                      [None, weights.shape[0]]),
        helper.make_tensor_value_info("classifier_product", TensorProto.FLOAT, [None, 2]),
        helper.make_tensor_value_info("classifier_logits", TensorProto.FLOAT, [None, 2]),
    ])
    replacement = [
        helper.make_node("MatMul", [classifier.input[0], "int8_classifier_weights"],
                         ["classifier_product"]),
        helper.make_node("Add", ["classifier_product", "int8_classifier_bias"],
                         ["classifier_logits"]),
        helper.make_node("Sigmoid", ["classifier_logits"], [classifier.output[1]]),
        helper.make_node("ArgMax", ["classifier_logits"], [classifier.output[0]],
                         axis=1, keepdims=0),
    ]
    nodes = []
    for node in graph.graph.node:
        nodes.extend(replacement if node == classifier else [node])
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    onnx.checker.check_model(graph)

    with tempfile.TemporaryDirectory() as directory:
        floating_path = Path(directory) / "classifier_fp32.onnx"
        quantized_path = Path(directory) / "classifier_int8.onnx"
        onnx.save(graph, floating_path)
        quantize_dynamic(str(floating_path), str(quantized_path),
                         weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
        quantized = onnx.load(quantized_path)
        onnx.checker.check_model(quantized)
        if not any(t.data_type == TensorProto.INT8 for t in quantized.graph.initializer):
            raise RuntimeError("quantization produced no INT8 weights")
        if not any(n.op_type == "MatMulInteger" for n in quantized.graph.node):
            raise RuntimeError("quantization produced no integer matrix multiplication")

        samples = np.array([["flood rescue"], ["a lovely sunny day"],
                            ["earthquake evacuation"], [""]], dtype=object)
        sessions = [rt.InferenceSession(str(p), providers=["CPUExecutionProvider"])
                    for p in (SOURCE, floating_path, quantized_path)]
        outputs = [s.run(None, {"text_input": samples}) for s in sessions]
        probabilities = [np.array([[p[0], p[1]] for p in o[1]]) for o in outputs]
        np.testing.assert_allclose(probabilities[0], probabilities[1], atol=1e-6)
        np.testing.assert_allclose(probabilities[0], probabilities[2], atol=0.02)
        np.testing.assert_array_equal(outputs[0][0], outputs[2][0])
        DESTINATION.write_bytes(quantized_path.read_bytes())
    print(f"wrote {DESTINATION}: verified INT8 weights and MatMulInteger inference")


if __name__ == "__main__":
    main()
