# Preprocessing and augmentation scripts

All commands are run from the repository root. Generated JSONL records use the
same `hash`, `text`, `entities` schema as the source data. Augmented records also
contain `_augmentation` provenance; the supplied baseline ignores extra fields.

## Apostrophe normalization

```bash
python data/normalize_apostrophes.py
```

Writes normalized train/dev datasets and reports to `data/preprocessing/`.
Only apostrophes inside alphanumeric words are normalized. External quote marks
and English-like contractions/possessives are preserved. Every replacement is
one Unicode character, so offsets do not shift.

## Cross-script substitutions

```bash
python data/augment_script_swap.py --input data/preprocessing/train.jsonl
```

Writes editor-friendly shards to `data/augmentation/script_swap/part-*.jsonl`
and the first 200 records to `data/augmentation/script_swap_preview.jsonl`.
A replacement is allowed only when both Cyrillic and Latin surfaces were
observed in train with the same label and have the same transliterated key. One
occurrence is changed per candidate. A single large file can still be requested
explicitly with `--output path/to/script_swap.jsonl`.

## Morphology

```bash
python data/augment_morphology.py --input data/preprocessing/train.jsonl
```

Writes `data/augmentation/morphology.jsonl`. Defaults require at least three
occurrences and label purity 1.0. Already inflected surfaces are conservatively
excluded. This is a candidate pool: subsample it during training.

## Controlled hard negatives

```bash
python data/augment_hard_negatives.py
```

Writes `data/augmentation/hard_negatives.jsonl`. These examples contain only
contextually unambiguous common words, generic roles/places, and non-entity
homonyms. Keep them a small part of training.

## Case augmentation

```bash
python data/augment_case.py --input data/preprocessing/train.jsonl
```

The default `per-record` mode creates at most one lower-case and one upper-case
copy per document. To produce the much larger maximum pool with one changed
entity per candidate, use:

```bash
python data/augment_case.py \
  --input data/preprocessing/train.jsonl \
  --mode per-entity \
  --output data/augmentation/case_per_entity.jsonl
```

Always keep the original real examples. Do not concatenate every generated pool
at full size without an ablation: synthetic/template data can dominate train and
reduce closed-test quality.
