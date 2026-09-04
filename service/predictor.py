from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


TAGS = (
    "O",
    "B-ORG",
    "I-ORG",
    "B-NAME",
    "I-NAME",
    "B-GEO",
    "I-GEO",
)
ENTITY_LABELS = {"ORG", "NAME", "GEO"}

JsonObject = dict[str, Any]
Offsets = list[tuple[int, int]]
ModelFeature = dict[str, list[int]]
Window = tuple[int, ModelFeature, Offsets]


def _default_model_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "model"


def _resolve_device() -> torch.device:
    requested = os.getenv("NER_DEVICE", "auto").lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("NER_DEVICE=cuda, but CUDA is not available")

    if requested == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            raise RuntimeError("NER_DEVICE=mps, but MPS is not available")

    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("NER_DEVICE must be one of: auto, cpu, cuda, mps")

    return torch.device(requested)


def _load_window_config(model_dir: Path) -> tuple[int, int]:
    path = model_dir / "baseline_config.json"
    if not path.exists():
        return 512, 128

    payload = json.loads(path.read_text(encoding="utf-8"))
    max_length = int(payload.get("max_length", 512))
    stride = int(payload.get("stride", 128))
    return max_length, stride


def _validate_window(tokenizer: Any, max_length: int, stride: int) -> None:
    content_length = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if content_length < 1:
        raise ValueError("max_length is too small for tokenizer special tokens")
    if not 0 <= stride < content_length:
        raise ValueError(f"stride must be between 0 and {content_length - 1}")


def _tokenize_windows(
    tokenizer: Any,
    text: str,
    *,
    max_length: int,
    stride: int,
) -> list[tuple[ModelFeature, Offsets]]:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
    )

    input_chunks = encoded["input_ids"]
    offset_chunks = encoded["offset_mapping"]

    if input_chunks and isinstance(input_chunks[0], int):
        input_chunks = [input_chunks]
        offset_chunks = [offset_chunks]

    windows: list[tuple[ModelFeature, Offsets]] = []

    for chunk_index, offsets in enumerate(offset_chunks):
        feature: ModelFeature = {}

        for key in ("input_ids", "attention_mask", "token_type_ids"):
            if key not in encoded:
                continue
            values = encoded[key]
            if values and isinstance(values[0], list):
                feature[key] = values[chunk_index]
            else:
                feature[key] = values

        windows.append(
            (
                feature,
                [(int(start), int(end)) for start, end in offsets],
            )
        )

    return windows


def _decode_bio_tokens(tokens: list[tuple[int, int, str]]) -> list[JsonObject]:
    entities: list[JsonObject] = []
    current: JsonObject | None = None

    def flush() -> None:
        nonlocal current
        if current is not None:
            entities.append(current)
            current = None

    for start, end, tag in tokens:
        if tag == "O":
            flush()
            continue

        prefix, separator, label = tag.partition("-")
        if separator != "-" or prefix not in {"B", "I"} or label not in ENTITY_LABELS:
            raise ValueError(f"Unsupported model tag: {tag!r}")

        if prefix == "B" or current is None or current["label"] != label:
            flush()
            current = {"label": label, "start": start, "end": end}
        else:
            current["end"] = max(current["end"], end)

    flush()
    return entities


class Predictor:
    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        batch_size: int = 16,
    ) -> None:
        self.model_dir = Path(
            model_dir or os.getenv("NER_MODEL_DIR") or _default_model_dir()
        ).expanduser().resolve()

        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        self.batch_size = int(os.getenv("NER_BATCH_SIZE", batch_size))
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")

        self.device = _resolve_device()
        self.max_length, self.stride = _load_window_config(self.model_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            use_fast=True,
            local_files_only=True,
        )
        if not self.tokenizer.is_fast:
            raise ValueError("Fast tokenizer is required for offset_mapping")

        _validate_window(self.tokenizer, self.max_length, self.stride)

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_dir,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()

        self.id2label = {
            int(index): str(label)
            for index, label in self.model.config.id2label.items()
        }
        if set(self.id2label.values()) != set(TAGS):
            raise ValueError(
                f"Model labels mismatch. Expected {list(TAGS)}, "
                f"got {self.id2label}"
            )

        # Не допускаем параллельные forward одного model instance.
        self._lock = threading.Lock()

        print(
            f"NER model loaded: {self.model_dir} | "
            f"device={self.device} | "
            f"max_length={self.max_length} | stride={self.stride}"
        )

    def _build_windows(self, records: list[JsonObject]) -> list[Window]:
        windows: list[Window] = []

        for record_index, record in enumerate(records):
            for feature, offsets in _tokenize_windows(
                self.tokenizer,
                record["text"],
                max_length=self.max_length,
                stride=self.stride,
            ):
                windows.append((record_index, feature, offsets))

        return windows

    @torch.inference_mode()
    def _predict_scores(
        self,
        windows: list[Window],
        record_count: int,
    ) -> list[dict[tuple[int, int], tuple[torch.Tensor, int]]]:
        aggregated: list[
            dict[tuple[int, int], tuple[torch.Tensor, int]]
        ] = [{} for _ in range(record_count)]

        for batch_start in range(0, len(windows), self.batch_size):
            batch_windows = windows[batch_start : batch_start + self.batch_size]

            batch = self.tokenizer.pad(
                [feature for _, feature, _ in batch_windows],
                padding=True,
                return_tensors="pt",
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}

            probabilities = torch.softmax(
                self.model(**batch).logits.float(),
                dim=-1,
            ).cpu()

            for row_index, (record_index, _, offsets) in enumerate(batch_windows):
                record_scores = aggregated[record_index]

                for token_index, (start, end) in enumerate(offsets):
                    # Special tokens имеют offset (0, 0).
                    if start == end:
                        continue

                    key = (start, end)
                    score = probabilities[row_index, token_index]

                    if key in record_scores:
                        previous, count = record_scores[key]
                        record_scores[key] = (previous + score, count + 1)
                    else:
                        record_scores[key] = (score.clone(), 1)

        return aggregated

    def predict(self, records: list[JsonObject]) -> list[JsonObject]:
        if not records:
            return []

        # text намеренно никак не strip/lower/normalize:
        # координаты должны относиться к исходной строке.
        windows = self._build_windows(records)

        with self._lock:
            scores = self._predict_scores(windows, len(records))

        predictions: list[JsonObject] = []

        for record, record_scores in zip(records, scores, strict=True):
            tagged_tokens: list[tuple[int, int, str]] = []

            for (start, end), (score_sum, count) in sorted(record_scores.items()):
                label_id = int((score_sum / count).argmax().item())
                tagged_tokens.append(
                    (start, end, self.id2label[label_id])
                )

            predictions.append(
                {
                    "hash": record["hash"],
                    "entities": _decode_bio_tokens(tagged_tokens),
                }
            )

        return predictions
