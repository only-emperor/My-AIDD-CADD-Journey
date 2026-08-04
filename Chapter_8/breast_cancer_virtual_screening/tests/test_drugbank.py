from __future__ import annotations

from pathlib import Path

from bcvs.sources.drugbank import iter_drugbank_xml


def test_drugbank_xml_stream(tmp_path: Path) -> None:
    xml = tmp_path / "drugbank.xml"
    xml.write_text(
        """<?xml version='1.0' encoding='UTF-8'?>
<drugbank xmlns='http://www.drugbank.ca'>
  <drug type='small molecule'>
    <drugbank-id primary='true'>DB00001</drugbank-id>
    <name>Example</name>
    <groups><group>approved</group></groups>
    <calculated-properties>
      <property><kind>SMILES</kind><value>CCO</value></property>
      <property><kind>InChIKey</kind><value>LFQSCWFLJHTTHZ-UHFFFAOYSA-N</value></property>
    </calculated-properties>
    <targets><target><name>Example target</name><polypeptide><gene-name>ESR1</gene-name></polypeptide></target></targets>
  </drug>
</drugbank>
""",
        encoding="utf-8",
    )
    rows = list(iter_drugbank_xml(xml))
    assert len(rows) == 1
    assert rows[0]["source_id"] == "DB00001"
    assert rows[0]["smiles"] == "CCO"
    assert rows[0]["drugbank_targets"] == "ESR1"
