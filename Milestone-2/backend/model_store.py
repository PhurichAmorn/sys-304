"""Versioned ONNX model store shared by the API and the retraining pipeline.

Layout under ``MODEL_ROOT`` (default ``/models``)::

    versions/v0001/model.onnx
    versions/v0001/manifest.json
    current.json          <- single source of truth for what is live

``current.json`` is replaced atomically with ``os.replace``, so a reader either
sees the whole previous pointer or the whole new one, never a partial write.

The API polls ``current.json`` and swaps its inference session when the version
changes. Promotion and rollback are both pointer writes, which is what makes
rolling back a bad model a single operation instead of a rebuild.
"""

import json
import os
import shutil
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path


def _resolve_model_root():
    """Pick a writable model root.

    ``/models`` is a Docker volume mount, so it only exists inside a container.
    Outside one (local API run, ``pytest``, a laptop) that path is either absent
    or unwritable, which used to make startup fail on the first write. Fall back
    to a directory beside the backend instead of crashing.
    """
    configured = Path(os.environ.get("MODEL_ROOT", "/models"))
    try:
        configured.mkdir(parents=True, exist_ok=True)
        probe = configured / ".write-probe"
        probe.touch()
        probe.unlink()
        return configured
    except OSError:
        local = Path(__file__).resolve().parent.parent / "models"
        local.mkdir(parents=True, exist_ok=True)
        return local


MODEL_ROOT = _resolve_model_root()
POINTER_NAME = "current.json"
MODEL_FILE_NAME = "model.onnx"
MANIFEST_NAME = "manifest.json"
BACKEND_DIR = Path(__file__).resolve().parent
IMAGE_FALLBACK = BACKEND_DIR / "model" / "model_fp32.onnx"


def pointer_path(root=None):
    return Path(root or MODEL_ROOT) / POINTER_NAME


def version_dir(version, root=None):
    return Path(root or MODEL_ROOT) / "versions" / version


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_pointer(root=None):
    path = pointer_path(root)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def write_pointer(pointer, root=None):
    """Atomically publish ``pointer`` as the live model version."""
    path = pointer_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w") as file:
            json.dump(pointer, file, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return pointer


def next_version(root=None):
    versions_dir = Path(root or MODEL_ROOT) / "versions"
    if not versions_dir.exists():
        return "v0001"
    highest = 0
    for entry in versions_dir.iterdir():
        if entry.is_dir() and entry.name.startswith("v"):
            try:
                highest = max(highest, int(entry.name[1:]))
            except ValueError:
                continue
    return f"v{highest + 1:04d}"


def export_to_onnx(vectorizer, model, destination):
    """Serialize a fitted TF-IDF + classifier pair as one ONNX graph.

    ``zipmap`` is disabled so the classifier returns a probability tensor
    instead of a per-row Python dict, which is cheaper on the serving path.
    """
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import StringTensorType
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([("tfidf", vectorizer), ("clf", model)])
    classifier = pipeline.steps[-1][1]
    graph = convert_sklearn(
        pipeline,
        initial_types=[("text_input", StringTensorType([None, 1]))],
        options={id(classifier): {"zipmap": False}},
    )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp_dir:
        staged = Path(temp_dir) / "model.onnx"
        staged.write_bytes(graph.SerializeToString())
        os.replace(staged, destination)
    return destination


def publish_version(
    *,
    vectorizer,
    model,
    metrics=None,
    source="retraining",
    previous_version=None,
    root=None,
    version=None,
):
    """Write a new model version then atomically promote it to live.

    The candidate is staged in a temporary directory on the same filesystem and
    moved into place with ``os.replace``, so a crash mid-write can never leave a
    half-written version directory pointed at by ``current.json``.
    """
    root = Path(root or MODEL_ROOT)
    current = read_pointer(root) or {}
    if previous_version is None:
        # Recorded here rather than left to the caller: rollback works only by
        # following this pointer, so a caller that forgets to pass it would
        # silently publish a version that cannot be rolled back.
        previous_version = current.get("version")
    version = version or next_version(root)
    if version == previous_version:
        raise ValueError(f"version {version} is already live; nothing to promote")
    destination = version_dir(version, root)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        shutil.rmtree(destination)
    staging = Path(tempfile.mkdtemp(dir=destination.parent, prefix=".staging-"))
    try:
        export_to_onnx(vectorizer, model, staging / MODEL_FILE_NAME)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    manifest = {
        "version": version,
        "promoted_at": utc_now(),
        "source": source,
        "model_file": MODEL_FILE_NAME,
        "metrics": metrics or {},
        "previous_version": previous_version,
    }
    (destination / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    write_pointer(manifest, root)
    return manifest


def rollback(root=None):
    """Repoint ``current.json`` at the version live before the current one."""
    pointer = read_pointer(root)
    if not pointer:
        raise RuntimeError("no model pointer to roll back from")
    target = pointer.get("previous_version")
    if not target:
        raise RuntimeError(f"{pointer.get('version')} has no previous version recorded")

    target_dir = version_dir(target, root)
    manifest_path = target_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise RuntimeError(f"previous version {target} is missing from the store")

    restored = json.loads(manifest_path.read_text())
    restored["rolled_back_from"] = pointer.get("version")
    restored["rolled_back_at"] = utc_now()
    restored["previous_version"] = pointer.get("version")
    write_pointer(restored, root)
    return restored


def bootstrap(root=None, fallback=None):
    """Seed version 1 from the ONNX artifact baked into the API image.

    Runs on first boot when the shared volume is empty, so the stack works with
    the pre-existing Milestone 3 model without any manual step.
    """
    root = Path(root or MODEL_ROOT)
    existing = read_pointer(root)
    if existing:
        return existing

    fallback = Path(fallback or IMAGE_FALLBACK)
    if not fallback.exists():
        raise RuntimeError(
            f"cannot bootstrap model store: no pointer and no image fallback at {fallback}"
        )

    version = "v0001"
    destination = version_dir(version, root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=destination.parent, prefix=".staging-"))
    try:
        shutil.copy2(fallback, staging / MODEL_FILE_NAME)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    manifest = {
        "version": version,
        "promoted_at": utc_now(),
        "source": "image-bootstrap",
        "model_file": MODEL_FILE_NAME,
        "metrics": {},
        "previous_version": None,
    }
    (destination / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    write_pointer(manifest, root)
    return manifest


def load_session(pointer, root=None):
    """Build an onnxruntime session for the version named by ``pointer``."""
    import onnxruntime as rt

    model_path = version_dir(pointer["version"], root) / pointer.get(
        "model_file", MODEL_FILE_NAME
    )
    if not model_path.exists():
        raise FileNotFoundError(f"model file missing for {pointer['version']}: {model_path}")

    options = rt.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    return rt.InferenceSession(str(model_path), sess_options=options)


def positive_probability(session, text):
    """Disaster-class probability for one string.

    Handles both ONNX output conventions: the ``probabilities`` tensor produced
    when zipmap is disabled, and the per-row class/probability dict produced by
    the ZipMap converter used for the image-bootstrapped artifact.
    """
    import numpy as np

    names = {output.name for output in session.get_outputs()}
    if "probabilities" in names:
        result = session.run(["probabilities"], {"text_input": np.array([[text]], dtype=object)})
        return float(result[0][0][1])
    if "output_probability" in names:
        result = session.run(
            ["output_probability"], {"text_input": np.array([[text]], dtype=object)}
        )
        return float(result[0][0][1])
    raise RuntimeError(f"unexpected ONNX model outputs: {sorted(names)}")


class ModelRegistry:
    """Thread-safe holder for the live inference session.

    Readers take the current session reference once and run to completion, so a
    promotion that lands mid-request never interrupts an in-flight prediction.
    """

    def __init__(self, root=None):
        self.root = Path(root or MODEL_ROOT)
        self._lock = threading.Lock()
        self._session = None
        self.pointer = None

    def start(self, fallback=None):
        with self._lock:
            pointer = bootstrap(self.root, fallback)
            self._session = load_session(pointer, self.root)
            self.pointer = pointer
        return self.pointer

    def session(self):
        with self._lock:
            return self._session

    def info(self):
        with self._lock:
            if self.pointer is None:
                return {"status": "not_loaded"}
            return {
                "status": "ok",
                "version": self.pointer.get("version"),
                "promoted_at": self.pointer.get("promoted_at"),
                "source": self.pointer.get("source"),
                "metrics": self.pointer.get("metrics", {}),
                "previous_version": self.pointer.get("previous_version"),
                "rolled_back_from": self.pointer.get("rolled_back_from"),
            }

    def poll(self):
        """Reload if the pointer moved. Returns True when a swap happened."""
        pointer = read_pointer(self.root)
        if not pointer:
            return False
        with self._lock:
            if self.pointer and pointer.get("version") == self.pointer.get(
                "version"
            ) and pointer.get("promoted_at") == self.pointer.get("promoted_at"):
                return False
            session = load_session(pointer, self.root)
            self._session = session
            self.pointer = pointer
        return True