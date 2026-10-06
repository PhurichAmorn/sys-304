"""Label disaster text through Ollama or an OpenAI-compatible chat API."""

import json
import urllib.error
import urllib.request

DEFAULT_PROVIDER = "ollama"
DEFAULT_BASE_URL = "http://host.docker.internal:11434"
DEFAULT_MODEL = "qwen2.5:0.5b"


def _endpoint(base_url, provider):
    suffix = "/api/chat" if provider == "ollama" else "/chat/completions"
    return base_url.rstrip("/") + suffix


def _parse_label(content):
    try:
        result = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("LLM response was not valid JSON") from error

    label = result.get("label")
    confidence = result.get("confidence")
    if isinstance(label, bool) or label not in (0, 1):
        raise ValueError(f"LLM returned invalid label: {label!r}; expected 0 or 1")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise TypeError("LLM returned invalid confidence")
    if not 0 <= float(confidence) <= 1:
        raise ValueError(f"LLM confidence is outside [0, 1]: {confidence!r}")
    return int(label), float(confidence)


def label_text(text, *, api_key, base_url=DEFAULT_BASE_URL, model=DEFAULT_MODEL,
               provider=DEFAULT_PROVIDER, timeout=30):
    if not text.strip():
        raise ValueError("empty text")
    system_prompt = (
        "Classify disaster-related text. Return JSON only with exactly two fields: "
        "label (integer 0 or 1) and confidence (number from 0 to 1). Use label 1 "
        "for a real disaster event and 0 for an ordinary or unrelated message. "
        "Never return any other label."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        "stream": False,
    }
    if provider == "ollama":
        payload["format"] = {
            "type": "object",
            "properties": {
                "label": {"type": "integer", "enum": [0, 1]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["label", "confidence"],
            "additionalProperties": False,
        }
        payload["options"] = {"temperature": 0}
    else:
        payload["temperature"] = 0
        payload["response_format"] = {"type": "json_object"}
    request = urllib.request.Request(
        _endpoint(base_url, provider),
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read())
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"LLM request failed: {error}") from error
    try:
        content = (
            body["message"]["content"]
            if provider == "ollama"
            else body["choices"][0]["message"]["content"]
        )
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("LLM response had no chat completion content") from error
    return _parse_label(content)


def label_rows(rows, *, api_key, base_url=DEFAULT_BASE_URL, model=DEFAULT_MODEL,
               provider=DEFAULT_PROVIDER, min_confidence=0.75, timeout=30):
    labeled = []
    for row in rows:
        text = row[1].strip()
        try:
            label, confidence = label_text(
                text, api_key=api_key, base_url=base_url, model=model,
                provider=provider, timeout=timeout
            )
        except (TypeError, ValueError, RuntimeError) as error:
            labeled.append({"text": text, "error": str(error)})
            continue
        if confidence >= min_confidence:
            labeled.append({"text": text, "label": label, "confidence": confidence})
    return labeled
