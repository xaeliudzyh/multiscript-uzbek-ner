"""Shared, dependency-free helpers for NER data augmentation scripts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


LABELS = {"ORG", "NAME", "GEO"}
JsonObject = dict[str, Any]


def read_jsonl(path: Path) -> list[JsonObject]:
    records: list[JsonObject] = []
    seen_hashes: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: empty line")
            record = json.loads(line)
            validate_record(record, f"{path}:{line_number}")
            if record["hash"] in seen_hashes:
                raise ValueError(f"{path}:{line_number}: duplicate hash")
            seen_hashes.add(record["hash"])
            records.append(record)
    return records


def validate_record(record: JsonObject, source: str = "record") -> None:
    if not isinstance(record, dict):
        raise ValueError(f"{source}: record must be an object")
    if not isinstance(record.get("hash"), str) or not record["hash"]:
        raise ValueError(f"{source}: hash must be a non-empty string")
    text = record.get("text")
    entities = record.get("entities")
    if not isinstance(text, str) or not isinstance(entities, list):
        raise ValueError(f"{source}: text/entities have invalid types")

    ordered: list[tuple[int, int, str]] = []
    seen: set[tuple[str, int, int]] = set()
    for index, entity in enumerate(entities):
        if not isinstance(entity, dict):
            raise ValueError(f"{source}/entities[{index}]: expected object")
        label, start, end = entity.get("label"), entity.get("start"), entity.get("end")
        if label not in LABELS:
            raise ValueError(f"{source}/entities[{index}]: invalid label {label!r}")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not 0 <= start < end <= len(text)
        ):
            raise ValueError(f"{source}/entities[{index}]: invalid offsets")
        key = (label, start, end)
        if key in seen:
            raise ValueError(f"{source}/entities[{index}]: duplicate entity")
        seen.add(key)
        ordered.append((start, end, label))

    ordered.sort()
    for left, right in zip(ordered, ordered[1:]):
        if right[0] < left[1]:
            raise ValueError(f"{source}: overlapping entities")


def stable_hash(kind: str, source_hash: str, details: Any) -> str:
    payload = json.dumps(details, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(f"{kind}\0{source_hash}\0{payload}".encode("utf-8")).hexdigest()
    return f"aug-{kind}-{digest[:32]}"


def replace_entities(
    record: JsonObject,
    replacements: Mapping[int, str],
    *,
    kind: str,
    details: JsonObject,
) -> JsonObject:
    """Replace selected entity surfaces and recalculate every following offset."""

    indexed = sorted(enumerate(record["entities"]), key=lambda item: item[1]["start"])
    text = record["text"]
    chunks: list[str] = []
    new_entities: list[JsonObject] = []
    cursor = 0
    output_length = 0

    for original_index, entity in indexed:
        start, end = entity["start"], entity["end"]
        prefix = text[cursor:start]
        chunks.append(prefix)
        output_length += len(prefix)

        surface = replacements.get(original_index, text[start:end])
        if not isinstance(surface, str) or not surface:
            raise ValueError("replacement surface must be a non-empty string")
        new_start = output_length
        chunks.append(surface)
        output_length += len(surface)
        new_entities.append(
            {"label": entity["label"], "start": new_start, "end": output_length}
        )
        cursor = end

    chunks.append(text[cursor:])
    new_text = "".join(chunks)
    payload = {
        "hash": stable_hash(kind, record["hash"], details),
        "text": new_text,
        "entities": new_entities,
        "_augmentation": {"type": kind, "source_hash": record["hash"], **details},
    }
    validate_record(payload, payload["hash"])
    return payload


def synthetic_record(
    text: str,
    entities: list[JsonObject],
    *,
    kind: str,
    details: JsonObject,
) -> JsonObject:
    payload = {
        "hash": stable_hash(kind, "synthetic", {"text": text, **details}),
        "text": text,
        "entities": entities,
        "_augmentation": {"type": kind, "source_hash": None, **details},
    }
    validate_record(payload, payload["hash"])
    return payload


def unique_records(records: Iterable[JsonObject]) -> Iterator[JsonObject]:
    seen: set[tuple[str, tuple[tuple[str, int, int], ...]]] = set()
    for record in records:
        key = (
            record["text"],
            tuple(
                (entity["label"], entity["start"], entity["end"])
                for entity in record["entities"]
            ),
        )
        if key in seen:
            continue
        seen.add(key)
        yield record


def write_jsonl(path: Path, records: Iterable[JsonObject]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    seen_hashes: set[str] = set()
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            validate_record(record)
            if record["hash"] in seen_hashes:
                raise ValueError(f"duplicate generated hash: {record['hash']}")
            seen_hashes.add(record["hash"])
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
            count += 1
    return count


def write_report(path: Path, report: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
