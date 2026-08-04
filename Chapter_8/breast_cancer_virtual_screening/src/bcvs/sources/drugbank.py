from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

import pandas as pd

from bcvs.utils import batched_iterable, write_dataframe

LOGGER = logging.getLogger(__name__)


def _text(element: ET.Element | None) -> str | None:
    return element.text.strip() if element is not None and element.text else None


def iter_drugbank_xml(xml_path: str | Path) -> Iterator[dict[str, object]]:
    """Stream an XML export supplied under the user's DrugBank license."""
    xml_path = Path(xml_path)
    context = ET.iterparse(xml_path, events=("start", "end"))
    _, root = next(context)
    namespace = ""
    if root.tag.startswith("{"):
        namespace = root.tag.split("}")[0] + "}"

    for event, elem in context:
        if event != "end" or elem.tag != f"{namespace}drug":
            continue
        primary_id = None
        for id_elem in elem.findall(f"{namespace}drugbank-id"):
            if id_elem.attrib.get("primary") == "true":
                primary_id = _text(id_elem)
                break
        if primary_id is None:
            primary_id = _text(elem.find(f"{namespace}drugbank-id"))
        name = _text(elem.find(f"{namespace}name"))
        groups = [
            _text(x) for x in elem.findall(f"{namespace}groups/{namespace}group") if _text(x)
        ]
        smiles = None
        inchikey = None
        for prop in elem.findall(f"{namespace}calculated-properties/{namespace}property"):
            kind = _text(prop.find(f"{namespace}kind"))
            value = _text(prop.find(f"{namespace}value"))
            if kind in {"SMILES", "Canonical SMILES"} and value:
                smiles = value
            elif kind == "InChIKey" and value:
                inchikey = value
        targets = []
        for target in elem.findall(f"{namespace}targets/{namespace}target"):
            gene = _text(target.find(f"{namespace}polypeptide/{namespace}gene-name"))
            target_name = _text(target.find(f"{namespace}name"))
            if gene or target_name:
                targets.append(gene or target_name)
        yield {
            "source": "DrugBank",
            "source_id": primary_id,
            "name": name,
            "smiles": smiles,
            "inchikey": inchikey,
            "drugbank_groups": ";".join(groups),
            "drugbank_targets": ";".join(dict.fromkeys(targets)),
        }
        elem.clear()
        root.clear()


def import_drugbank_xml(
    xml_path: str | Path,
    output_path: str | Path,
    batch_size: int = 10000,
) -> Path:
    """Write a licensed DrugBank XML export to a normalized CSV.GZ/Parquet table."""
    output_path = Path(output_path)
    parts_dir = output_path.parent / f"{output_path.stem}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    for idx, batch in enumerate(batched_iterable(iter_drugbank_xml(xml_path), batch_size)):
        part = write_dataframe(pd.DataFrame(batch), parts_dir / f"part_{idx:05d}.csv.gz")
        parts.append(part)
        LOGGER.info("Parsed DrugBank XML part %d (%d rows)", idx, len(batch))
    frames = [pd.read_csv(p) for p in parts]
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return write_dataframe(merged, output_path)
