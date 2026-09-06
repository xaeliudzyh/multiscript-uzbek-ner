from __future__ import annotations

import json
import os
import threading
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, PreTrainedTokenizerBase


JsonObject = dict[str, Any]
ModelFeature = dict[str, list[int]]
Offsets = list[tuple[int, int]]
Window = tuple[int, ModelFeature, Offsets]

ENTITY_LABELS = ("ORG", "NAME", "GEO")
ENTITY_CHARS = frozenset("'’ʻʼ‘`-")
APOSTROPHES = "'’ʻʼ‘`"

TAGS = (
    "O",
    "B-ORG",
    "I-ORG",
    "B-NAME",
    "I-NAME",
    "B-GEO",
    "I-GEO",
)
TAG_TO_ID = {tag: i for i, tag in enumerate(TAGS)}
ID_TO_TAG = {i: tag for tag, i in TAG_TO_ID.items()}

MODEL_NAME = os.getenv(
    "NER_MODEL_NAME",
    "dashakoryakovskaya/uzbek-ner",
)
TOKENIZER_NAME = os.getenv(
    "NER_TOKENIZER_NAME",
    "FacebookAI/xlm-roberta-large",
)

DEFAULT_MAX_LENGTH = 512
DEFAULT_STRIDE = 128
DEFAULT_BATCH_SIZE = 16


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_train_path() -> Path:
    return _project_root() / "data" / "train.jsonl"


def _resolve_device() -> torch.device:
    requested = os.getenv("NER_DEVICE", "auto").strip().lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("NER_DEVICE=cuda, but CUDA is not available")
        return torch.device("cuda")

    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("NER_DEVICE=mps, but MPS is not available")
        return torch.device("mps")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError("NER_DEVICE must be one of: auto, cpu, cuda, mps")


def read_jsonl(path: Path) -> list[JsonObject]:
    records: list[JsonObject] = []
    seen_hashes: set[str] = set()

    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: empty line")

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {error}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}:{line_number}: record must be an object"
                )

            record_hash = record.get("hash")
            if not isinstance(record_hash, str) or not record_hash:
                raise ValueError(
                    f"{path}:{line_number}: hash must be a non-empty string"
                )

            if record_hash in seen_hashes:
                raise ValueError(
                    f"{path}:{line_number}: duplicate hash {record_hash}"
                )

            seen_hashes.add(record_hash)
            records.append(record)

    if not records:
        raise ValueError(f"{path}: no records")

    return records


# ============================================================
# Tokenization / model inference
# This follows the supplied predict.ipynb logic:
# - xlm-roberta-large tokenizer
# - manual overlapping windows
# - probability averaging for tokens repeated in overlapping windows
# - BIO exact-span decoding
# ============================================================

def tokenize_windows(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    *,
    max_length: int,
    stride: int,
) -> list[tuple[ModelFeature, Offsets]]:
    content_length = (
        max_length
        - tokenizer.num_special_tokens_to_add(pair=False)
    )

    if content_length < 1:
        raise ValueError(
            "max_length is too small for tokenizer special tokens"
        )

    if not 0 <= stride < content_length:
        raise ValueError(
            f"stride must be between 0 and {content_length - 1}"
        )

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
        verbose=False,
    )

    input_ids = [
        int(token_id)
        for token_id in encoded["input_ids"]
    ]
    offsets = [
        (int(start), int(end))
        for start, end in encoded["offset_mapping"]
    ]

    if len(input_ids) != len(offsets):
        raise RuntimeError(
            "tokenizer returned different input_ids and offset_mapping lengths"
        )

    windows: list[tuple[ModelFeature, Offsets]] = []
    step = content_length - stride
    window_starts = (
        range(0, len(input_ids), step)
        if input_ids
        else (0,)
    )

    for window_start in window_starts:
        window_end = min(
            window_start + content_length,
            len(input_ids),
        )

        content_ids = input_ids[window_start:window_end]
        content_offsets = offsets[window_start:window_end]

        prefix_id = (
            tokenizer.cls_token_id
            if tokenizer.cls_token_id is not None
            else tokenizer.bos_token_id
        )
        suffix_id = (
            tokenizer.sep_token_id
            if tokenizer.sep_token_id is not None
            else tokenizer.eos_token_id
        )

        prefix = (
            [int(prefix_id)]
            if prefix_id is not None
            else []
        )
        suffix = (
            [int(suffix_id)]
            if suffix_id is not None
            else []
        )

        if (
            len(prefix) + len(suffix)
            != tokenizer.num_special_tokens_to_add(pair=False)
        ):
            raise ValueError(
                "unsupported single-sequence special-token layout"
            )

        window_input_ids = prefix + content_ids + suffix
        window_offsets = (
            [(0, 0)] * len(prefix)
            + content_offsets
            + [(0, 0)] * len(suffix)
        )

        feature: ModelFeature = {
            "input_ids": window_input_ids,
            "attention_mask": [1] * len(window_input_ids),
        }

        if "token_type_ids" in tokenizer.model_input_names:
            feature["token_type_ids"] = [
                0
            ] * len(window_input_ids)

        windows.append((feature, window_offsets))

        if window_end >= len(input_ids):
            break

    return windows


def _build_windows(
    records: list[JsonObject],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
    stride: int,
) -> list[Window]:
    windows: list[Window] = []

    for record_index, record in enumerate(records):
        for feature, offsets in tokenize_windows(
            tokenizer,
            record["text"],
            max_length=max_length,
            stride=stride,
        ):
            windows.append(
                (record_index, feature, offsets)
            )

    return windows


def decode_bio_tokens(
    tagged_tokens: list[
        tuple[int, int, str, float]
    ],
) -> list[JsonObject]:
    entities: list[JsonObject] = []
    current: JsonObject | None = None
    token_confidences: list[float] = []

    def flush() -> None:
        nonlocal current, token_confidences

        if current is not None:
            current["confidence"] = (
                sum(token_confidences)
                / len(token_confidences)
                if token_confidences
                else None
            )
            current.setdefault("source", "model")
            entities.append(current)

        current = None
        token_confidences = []

    for start, end, tag, confidence in tagged_tokens:
        if tag == "O":
            flush()
            continue

        prefix, separator, label = tag.partition("-")

        if (
            separator != "-"
            or prefix not in {"B", "I"}
            or label not in ENTITY_LABELS
        ):
            raise ValueError(
                f"Unsupported model tag: {tag!r}"
            )

        if (
            prefix == "B"
            or current is None
            or current["label"] != label
        ):
            flush()
            current = {
                "label": label,
                "start": start,
                "end": end,
                "source": "model",
            }
            token_confidences = [confidence]
        else:
            current["end"] = max(
                int(current["end"]),
                end,
            )
            token_confidences.append(confidence)

    flush()

    best: dict[
        tuple[str, int, int],
        JsonObject,
    ] = {}

    for entity in entities:
        key = (
            str(entity["label"]),
            int(entity["start"]),
            int(entity["end"]),
        )

        old = best.get(key)
        old_conf = (
            float(old.get("confidence") or 0.0)
            if old is not None
            else -1.0
        )
        new_conf = float(
            entity.get("confidence") or 0.0
        )

        if old is None or new_conf > old_conf:
            best[key] = entity

    return sorted(
        best.values(),
        key=lambda e: (
            int(e["start"]),
            int(e["end"]),
            str(e["label"]),
        ),
    )


# ============================================================
# Train-only postprocessing from supplied predict.ipynb
# ============================================================

def is_entity_char(char: str) -> bool:
    return (
        char.isalnum()
        or char in ENTITY_CHARS
    )


def expand_to_attached_word(
    text: str,
    start: int,
    end: int,
) -> tuple[int, int]:
    while (
        start > 0
        and start < len(text)
        and is_entity_char(text[start - 1])
        and is_entity_char(text[start])
    ):
        start -= 1

    while (
        end > 0
        and end < len(text)
        and is_entity_char(text[end - 1])
        and is_entity_char(text[end])
    ):
        end += 1

    return start, end


def normalize_surface(surface: str) -> str:
    normalized = surface.casefold()

    for apostrophe in APOSTROPHES:
        normalized = normalized.replace(
            apostrophe,
            "'",
        )

    return " ".join(normalized.split())


def build_surface_lexicon(
    train_records: list[JsonObject],
) -> dict[str, Counter[str]]:
    lexicon: dict[
        str,
        Counter[str],
    ] = defaultdict(Counter)

    for record in train_records:
        text = record.get("text")
        entities = record.get("entities")

        if (
            not isinstance(text, str)
            or not isinstance(entities, list)
        ):
            raise ValueError(
                "train records must contain text and entities"
            )

        for entity in entities:
            label = entity.get("label")
            start = entity.get("start")
            end = entity.get("end")

            if (
                label not in ENTITY_LABELS
                or not isinstance(start, int)
                or not isinstance(end, int)
                or not 0 <= start < end <= len(text)
            ):
                raise ValueError(
                    "invalid train entity"
                )

            lexicon[
                text[start:end]
            ][str(label)] += 1

    return dict(lexicon)


def build_normalized_lexicon(
    exact_lexicon: dict[
        str,
        Counter[str],
    ],
) -> dict[str, Counter[str]]:
    result: dict[
        str,
        Counter[str],
    ] = defaultdict(Counter)

    for surface, counts in exact_lexicon.items():
        result[
            normalize_surface(surface)
        ].update(counts)

    return dict(result)


def lexicon_label(
    surface: str,
    lexicon: dict[
        str,
        Counter[str],
    ],
    *,
    min_support: int,
    min_purity: float,
) -> str | None:
    counts = lexicon.get(surface)

    if not counts:
        return None

    label, support = counts.most_common(1)[0]
    total = sum(counts.values())

    if (
        support < min_support
        or support / total < min_purity
    ):
        return None

    return label


def confident_label(
    counts: Counter[str] | None,
    *,
    min_support: int,
    min_purity: float,
) -> tuple[str, int] | None:
    if not counts:
        return None

    label, support = counts.most_common(1)[0]

    if (
        support < min_support
        or support / sum(counts.values())
        < min_purity
    ):
        return None

    return label, support


def _deduplicate_entities(
    entities: list[JsonObject],
) -> tuple[list[JsonObject], int]:
    best: dict[
        tuple[str, int, int],
        JsonObject,
    ] = {}

    for entity in entities:
        key = (
            str(entity["label"]),
            int(entity["start"]),
            int(entity["end"]),
        )

        old = best.get(key)

        if old is None:
            best[key] = entity
            continue

        old_conf = float(
            old.get("confidence") or 0.0
        )
        new_conf = float(
            entity.get("confidence") or 0.0
        )

        if new_conf > old_conf:
            best[key] = entity

    return (
        list(best.values()),
        len(entities) - len(best),
    )


def postprocess_predictions(
    input_records: list[JsonObject],
    predictions: list[JsonObject],
    exact_lexicon: dict[
        str,
        Counter[str],
    ],
    *,
    expand_word_boundaries: bool = True,
    relabel: bool = True,
    min_label_support: int = 2,
    min_label_purity: float = 0.9,
) -> tuple[
    list[JsonObject],
    Counter[str],
]:
    texts = {
        record["hash"]: record["text"]
        for record in input_records
    }

    result: list[JsonObject] = []
    stats: Counter[str] = Counter()

    for record in predictions:
        record_hash = record["hash"]
        text = texts[record_hash]
        corrected: list[JsonObject] = []

        for original in record["entities"]:
            entity = dict(original)

            label = str(entity["label"])
            start = int(entity["start"])
            end = int(entity["end"])

            if not 0 <= start < end <= len(text):
                raise ValueError(
                    f"prediction {record_hash}: "
                    f"invalid entity {entity!r}"
                )

            if expand_word_boundaries:
                new_start, new_end = (
                    expand_to_attached_word(
                        text,
                        start,
                        end,
                    )
                )

                if (
                    new_start,
                    new_end,
                ) != (
                    start,
                    end,
                ):
                    stats["boundary_changes"] += 1
                    start, end = (
                        new_start,
                        new_end,
                    )

            if relabel:
                new_label = lexicon_label(
                    text[start:end],
                    exact_lexicon,
                    min_support=min_label_support,
                    min_purity=min_label_purity,
                )

                if (
                    new_label is not None
                    and new_label != label
                ):
                    label = new_label
                    stats["label_changes"] += 1

            entity.update({
                "label": label,
                "start": start,
                "end": end,
            })

            corrected.append(entity)

        unique, removed = _deduplicate_entities(
            corrected
        )
        stats["duplicates_removed"] += removed

        result.append({
            "hash": record_hash,
            "entities": sorted(
                unique,
                key=lambda e: (
                    int(e["start"]),
                    int(e["end"]),
                    str(e["label"]),
                ),
            ),
        })

    return result, stats


def normalized_relabel(
    input_records: list[JsonObject],
    predictions: list[JsonObject],
    normalized_lexicon: dict[
        str,
        Counter[str],
    ],
    *,
    min_support: int,
    min_purity: float,
) -> tuple[
    list[JsonObject],
    Counter[str],
]:
    texts = {
        record["hash"]: record["text"]
        for record in input_records
    }

    result: list[JsonObject] = []
    stats: Counter[str] = Counter()

    for record in predictions:
        text = texts[record["hash"]]
        corrected: list[JsonObject] = []

        for original in record["entities"]:
            entity = dict(original)

            start = int(entity["start"])
            end = int(entity["end"])
            label = str(entity["label"])

            decision = confident_label(
                normalized_lexicon.get(
                    normalize_surface(
                        text[start:end]
                    )
                ),
                min_support=min_support,
                min_purity=min_purity,
            )

            if (
                decision is not None
                and decision[0] != label
            ):
                entity["label"] = decision[0]
                stats[
                    "normalized_label_changes"
                ] += 1

            corrected.append(entity)

        unique, removed = _deduplicate_entities(
            corrected
        )
        stats["duplicates_removed"] += removed

        result.append({
            "hash": record["hash"],
            "entities": unique,
        })

    return result, stats


def has_entity_boundaries(
    text: str,
    start: int,
    end: int,
) -> bool:
    left_is_inside_word = (
        start > 0
        and is_entity_char(text[start - 1])
        and is_entity_char(text[start])
    )

    right_is_inside_word = (
        end < len(text)
        and is_entity_char(text[end - 1])
        and is_entity_char(text[end])
    )

    return not (
        left_is_inside_word
        or right_is_inside_word
    )


def propagate_repeated_mentions(
    input_records: list[JsonObject],
    predictions: list[JsonObject],
    *,
    min_length: int,
) -> tuple[
    list[JsonObject],
    Counter[str],
]:
    texts = {
        record["hash"]: record["text"]
        for record in input_records
    }

    result: list[JsonObject] = []
    stats: Counter[str] = Counter()

    for record in predictions:
        text = texts[record["hash"]]
        entities = [
            dict(entity)
            for entity in record["entities"]
        ]
        seeds = list(entities)

        occupied = [
            (
                int(entity["start"]),
                int(entity["end"]),
            )
            for entity in entities
        ]

        for seed in seeds:
            surface = text[
                int(seed["start"]):
                int(seed["end"])
            ]

            if len(surface) < min_length:
                continue

            search_from = 0

            while True:
                start = text.find(
                    surface,
                    search_from,
                )

                if start < 0:
                    break

                end = start + len(surface)
                search_from = start + 1

                if not has_entity_boundaries(
                    text,
                    start,
                    end,
                ):
                    continue

                if any(
                    start < old_end
                    and end > old_start
                    for old_start, old_end
                    in occupied
                ):
                    continue

                propagated = dict(seed)
                propagated.update({
                    "start": start,
                    "end": end,
                    "source": "repeat_propagation",
                })

                entities.append(propagated)
                occupied.append((start, end))
                stats[
                    "repeated_mentions_added"
                ] += 1

        result.append({
            "hash": record["hash"],
            "entities": entities,
        })

    return result, stats


def resolve_overlaps(
    input_records: list[JsonObject],
    predictions: list[JsonObject],
    normalized_lexicon: dict[
        str,
        Counter[str],
    ],
) -> tuple[
    list[JsonObject],
    Counter[str],
]:
    texts = {
        record["hash"]: record["text"]
        for record in input_records
    }

    result: list[JsonObject] = []
    stats: Counter[str] = Counter()

    for record in predictions:
        text = texts[record["hash"]]

        def rank(
            entity: JsonObject,
        ) -> tuple[int, int, int, int]:
            surface = text[
                int(entity["start"]):
                int(entity["end"])
            ]

            counts = normalized_lexicon.get(
                normalize_surface(surface)
            )

            decision = confident_label(
                counts,
                min_support=2,
                min_purity=0.9,
            )

            known = int(
                decision is not None
                and decision[0]
                == entity["label"]
            )
            support = (
                decision[1]
                if known
                else 0
            )

            return (
                len(surface),
                known,
                support,
                -int(entity["start"]),
            )

        kept: list[JsonObject] = []

        for entity in sorted(
            record["entities"],
            key=rank,
            reverse=True,
        ):
            overlaps = any(
                int(entity["start"])
                < int(previous["end"])
                and int(entity["end"])
                > int(previous["start"])
                for previous in kept
            )

            if overlaps:
                stats[
                    "overlapping_entities_removed"
                ] += 1
            else:
                kept.append(entity)

        result.append({
            "hash": record["hash"],
            "entities": kept,
        })

    return result, stats


def filter_short_unknown_entities(
    input_records: list[JsonObject],
    predictions: list[JsonObject],
    exact_lexicon: dict[
        str,
        Counter[str],
    ],
    *,
    max_length: int,
) -> tuple[
    list[JsonObject],
    Counter[str],
]:
    texts = {
        record["hash"]: record["text"]
        for record in input_records
    }

    result: list[JsonObject] = []
    stats: Counter[str] = Counter()

    for record in predictions:
        text = texts[record["hash"]]
        kept: list[JsonObject] = []

        for entity in record["entities"]:
            surface = text[
                int(entity["start"]):
                int(entity["end"])
            ]

            known_with_label = (
                str(entity["label"])
                in exact_lexicon.get(
                    surface,
                    {},
                )
            )

            if (
                len(surface.strip()) <= max_length
                and not known_with_label
            ):
                stats[
                    "short_unknown_entities_removed"
                ] += 1
                continue

            kept.append(entity)

        result.append({
            "hash": record["hash"],
            "entities": kept,
        })

    return result, stats


def sort_and_deduplicate(
    predictions: list[JsonObject],
) -> list[JsonObject]:
    result: list[JsonObject] = []

    for record in predictions:
        unique, _ = _deduplicate_entities(
            record["entities"]
        )

        result.append({
            "hash": record["hash"],
            "entities": sorted(
                unique,
                key=lambda entity: (
                    int(entity["start"]),
                    int(entity["end"]),
                    str(entity["label"]),
                ),
            ),
        })

    return result


def run_postprocessing(
    input_records: list[JsonObject],
    predictions: list[JsonObject],
    exact_lexicon: dict[
        str,
        Counter[str],
    ],
    normalized_lexicon: dict[
        str,
        Counter[str],
    ],
) -> tuple[
    list[JsonObject],
    Counter[str],
]:
    stats: Counter[str] = Counter()

    result, stage_stats = postprocess_predictions(
        input_records,
        predictions,
        exact_lexicon,
        expand_word_boundaries=True,
        relabel=True,
        min_label_support=2,
        min_label_purity=0.9,
    )
    stats.update(stage_stats)

    result, stage_stats = normalized_relabel(
        input_records,
        result,
        normalized_lexicon,
        min_support=2,
        min_purity=1.0,
    )
    stats.update(stage_stats)

    result, stage_stats = propagate_repeated_mentions(
        input_records,
        result,
        min_length=4,
    )
    stats.update(stage_stats)

    result, stage_stats = resolve_overlaps(
        input_records,
        result,
        normalized_lexicon,
    )
    stats.update(stage_stats)

    result, stage_stats = filter_short_unknown_entities(
        input_records,
        result,
        exact_lexicon,
        max_length=2,
    )
    stats.update(stage_stats)

    result = sort_and_deduplicate(result)

    return result, stats


# ============================================================
# Predictor service object
# ============================================================

class Predictor:
    def __init__(
        self,
        *,
        train_path: str | Path | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
        stride: int = DEFAULT_STRIDE,
    ) -> None:
        self.device = _resolve_device()

        self.batch_size = int(
            os.getenv(
                "NER_BATCH_SIZE",
                str(batch_size),
            )
        )
        self.max_length = int(
            os.getenv(
                "NER_MAX_LENGTH",
                str(max_length),
            )
        )
        self.stride = int(
            os.getenv(
                "NER_STRIDE",
                str(stride),
            )
        )

        if self.batch_size < 1:
            raise ValueError(
                "NER_BATCH_SIZE must be positive"
            )

        self.train_path = Path(
            train_path
            or os.getenv("NER_TRAIN_PATH")
            or _default_train_path()
        ).expanduser().resolve()

        if not self.train_path.exists():
            raise FileNotFoundError(
                f"Train file not found: "
                f"{self.train_path}"
            )

        print(
            f"[NER] Reading train lexicon: "
            f"{self.train_path}"
        )

        train_records = read_jsonl(
            self.train_path
        )

        self.exact_lexicon = (
            build_surface_lexicon(
                train_records
            )
        )
        self.normalized_lexicon = (
            build_normalized_lexicon(
                self.exact_lexicon
            )
        )

        token = (
            os.getenv("HF_TOKEN")
            or None
        )

        print(
            f"[NER] Loading tokenizer: "
            f"{TOKENIZER_NAME}"
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                TOKENIZER_NAME,
                use_fast=True,
                token=token,
            )
        )

        if not self.tokenizer.is_fast:
            raise ValueError(
                "Fast tokenizer is required "
                "for offset_mapping"
            )

        print(
            f"[NER] Loading model: "
            f"{MODEL_NAME}"
        )

        self.model = (
            AutoModelForTokenClassification
            .from_pretrained(
                MODEL_NAME,
                num_labels=len(TAGS),
                id2label=ID_TO_TAG,
                label2id=TAG_TO_ID,
                token=token,
            )
            .to(self.device)
        )

        self.model.eval()

        self.use_amp = (
            self.device.type == "cuda"
        )

        self._lock = threading.Lock()

        print(
            "[NER] Ready | "
            f"device={self.device} | "
            f"max_length={self.max_length} | "
            f"stride={self.stride} | "
            f"batch_size={self.batch_size} | "
            f"train_records={len(train_records)} | "
            f"lexicon_surfaces={len(self.exact_lexicon)}"
        )

    @torch.inference_mode()
    def _predict_raw(
        self,
        records: list[JsonObject],
    ) -> list[JsonObject]:
        windows = _build_windows(
            records,
            self.tokenizer,
            max_length=self.max_length,
            stride=self.stride,
        )

        aggregated: list[
            dict[
                tuple[int, int],
                tuple[torch.Tensor, int],
            ]
        ] = [
            {}
            for _ in records
        ]

        for batch_start in range(
            0,
            len(windows),
            self.batch_size,
        ):
            batch_windows = windows[
                batch_start:
                batch_start + self.batch_size
            ]

            padded = self.tokenizer.pad(
                [
                    feature
                    for _, feature, _
                    in batch_windows
                ],
                padding=True,
                return_tensors="pt",
            )

            padded = {
                key: value.to(self.device)
                for key, value
                in padded.items()
            }

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_amp,
            ):
                logits = self.model(
                    **padded
                ).logits

            probabilities = torch.softmax(
                logits.float(),
                dim=-1,
            ).cpu()

            for row, (
                record_index,
                _,
                offsets,
            ) in enumerate(batch_windows):
                record_scores = aggregated[
                    record_index
                ]

                for token_index, (
                    start,
                    end,
                ) in enumerate(offsets):
                    if start == end:
                        continue

                    score = probabilities[
                        row,
                        token_index,
                    ]

                    old = record_scores.get(
                        (start, end)
                    )

                    if old is None:
                        record_scores[
                            (start, end)
                        ] = (
                            score.clone(),
                            1,
                        )
                    else:
                        previous, count = old
                        record_scores[
                            (start, end)
                        ] = (
                            previous + score,
                            count + 1,
                        )

        output: list[JsonObject] = []

        for record, scores in zip(
            records,
            aggregated,
        ):
            tagged: list[
                tuple[int, int, str, float]
            ] = []

            for (
                start,
                end,
            ), (
                score_sum,
                count,
            ) in sorted(
                scores.items()
            ):
                average_score = (
                    score_sum / count
                )

                tag_id = int(
                    average_score
                    .argmax()
                    .item()
                )

                tagged.append((
                    start,
                    end,
                    ID_TO_TAG[tag_id],
                    float(
                        average_score[
                            tag_id
                        ].item()
                    ),
                ))

            output.append({
                "hash": record["hash"],
                "entities": decode_bio_tokens(
                    tagged
                ),
            })

        return output

    def predict(
        self,
        records: list[JsonObject],
    ) -> tuple[
        list[JsonObject],
        Counter[str],
    ]:
        if not records:
            return [], Counter()

        with self._lock:
            raw_predictions = (
                self._predict_raw(records)
            )

        return run_postprocessing(
            records,
            raw_predictions,
            self.exact_lexicon,
            self.normalized_lexicon,
        )

    def info(self) -> dict[str, Any]:
        return {
            "model_name": MODEL_NAME,
            "tokenizer_name": TOKENIZER_NAME,
            "architecture": (
                "token-classification BIO"
            ),
            "device": str(self.device),
            "labels": list(
                ENTITY_LABELS
            ),
            "max_length": self.max_length,
            "stride": self.stride,
            "batch_size": self.batch_size,
            "train_path": str(
                self.train_path
            ),
            "lexicon_surfaces": len(
                self.exact_lexicon
            ),
            "postprocessing": [
                "attached-word boundary expansion",
                "exact train-lexicon relabel",
                "normalized apostrophe/case relabel",
                "repeated-mention propagation",
                "overlap resolution",
                "short unknown span filtering",
                "sort + deduplicate",
            ],
        }
