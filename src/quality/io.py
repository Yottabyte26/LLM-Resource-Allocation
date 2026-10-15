"""Read official compressed JSONL without mutating files or retaining text."""

from __future__ import annotations

import hashlib
import json
import lzma
from pathlib import Path
from typing import Iterator

import yaml


def load_field_aliases() -> dict[str, tuple[str, ...]]:
    """Build FIELD_ALIASES exclusively from the project's central configuration."""
    root = Path(__file__).resolve().parents[2]
    with (root / "configs" / "quality_metrics.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return {m["name"]: tuple(m["raw_aliases"]) for m in config["metrics"] if m.get("raw_aliases")}


FIELD_ALIASES = load_field_aliases()


def infer_source_domain(path: Path, record: dict) -> tuple[str | None, str]:
    """Resolve an existing domain or infer only documented extension prefixes.

    Returns (domain, provenance rule). Conflicting metadata raises rather than
    silently choosing a domain. Does not modify the input record.
    """
    inferred = next((d for d in ("arxiv", "github") if path.name.startswith(f"{d}_")), None)
    domain = record.get("_source_domain")
    if domain is not None and not isinstance(domain, str):
        raise ValueError(f"Invalid _source_domain type in {path.name}")
    if domain:
        if inferred and domain != inferred:
            raise ValueError(f"Conflicting domain in {path.name}: {domain} vs {inferred}")
        return domain, "record._source_domain"
    if inferred:
        return inferred, "filename_prefix"
    return None, "unresolved"


def resolve_metric_value(record: dict, metric: dict) -> tuple[object, str | None]:
    """Resolve a configured official name or explicit alias without source edits."""
    aliases = metric.get("raw_aliases", FIELD_ALIASES.get(metric["name"], ()))
    names = [name for name in [metric["name"], *aliases] if name in record]
    if not names:
        return None, None
    if len(names) > 1 and any(record[k] != record[names[0]] for k in names[1:]):
        raise ValueError(f"Conflicting raw aliases for {metric['name']}: {names}")
    return record[names[0]], names[0]


def discover_quality_files(root: Path) -> dict[str, list[Path]]:
    """Discover exact A1 and every actual A2/A3 shard; fail on absent input."""
    base = root / "real_attachments" / "A_data_value"
    extended = base / "slimpajama_quality_extended"
    datasets = {
        "A1": [base / "slimpajama_quality_signal_sample.jsonl.xz"],
        "A2": sorted(extended.glob("arxiv_*.jsonl.xz")),
        "A3": sorted(extended.glob("github_*.jsonl.xz")),
    }
    for dataset, files in datasets.items():
        if not files or any(not p.is_file() for p in files):
            raise FileNotFoundError(f"{dataset}: no readable files under {base}")
    return datasets


def sha256_file(path: Path) -> str:
    """Hash compressed source bytes in bounded memory for read-only verification."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict | None, str | None]]:
    """Yield physical line number, object or failure; never retain content."""
    with lzma.open(path, "rt", encoding="utf-8", errors="strict") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("JSON record is not an object")
                yield line_number, record, None
            except (json.JSONDecodeError, ValueError) as exc:
                # Do not copy raw lines: a failed record could contain huge text.
                yield line_number, None, str(exc)
