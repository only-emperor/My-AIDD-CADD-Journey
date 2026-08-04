from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


@dataclass
class StandardizedMolecule:
    input_smiles: str | None
    standardized_smiles: str | None = None
    inchikey: str | None = None
    valid: bool = False
    standardization_error: str | None = None
    atom_count: int | None = None
    heavy_atom_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MoleculeStandardizer:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.allowed_elements = set(cfg.get("allowed_elements", []))
        self.uncharger = rdMolStandardize.Uncharger()
        self.tautomer_enumerator = rdMolStandardize.TautomerEnumerator()

    def standardize(self, smiles: str | None) -> StandardizedMolecule:
        result = StandardizedMolecule(input_smiles=smiles)
        if smiles is None or not str(smiles).strip():
            result.standardization_error = "missing_smiles"
            return result
        try:
            mol = Chem.MolFromSmiles(str(smiles), sanitize=True)
            if mol is None:
                raise ValueError("RDKit could not parse SMILES")
            if self.cfg.get("cleanup", True):
                mol = rdMolStandardize.Cleanup(mol)
            if self.cfg.get("largest_fragment", True):
                mol = rdMolStandardize.FragmentParent(mol)
            if self.cfg.get("uncharge", True):
                mol = self.uncharger.uncharge(mol)
            if self.cfg.get("canonical_tautomer", False):
                mol = self.tautomer_enumerator.Canonicalize(mol)
            elements = {atom.GetSymbol() for atom in mol.GetAtoms()}
            disallowed = elements - self.allowed_elements if self.allowed_elements else set()
            if disallowed:
                raise ValueError(f"disallowed_elements:{','.join(sorted(disallowed))}")
            smiles_out = Chem.MolToSmiles(
                mol,
                canonical=True,
                isomericSmiles=self.cfg.get("isomeric_smiles", True),
            )
            result.standardized_smiles = smiles_out
            result.inchikey = Chem.MolToInchiKey(mol)
            result.valid = True
            result.atom_count = mol.GetNumAtoms()
            result.heavy_atom_count = mol.GetNumHeavyAtoms()
            return result
        except Exception as exc:  # noqa: BLE001 - row-level invalid molecules are expected
            result.standardization_error = str(exc)
            return result
