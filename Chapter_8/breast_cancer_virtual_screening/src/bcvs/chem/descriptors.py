from __future__ import annotations

from typing import Any

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

try:
    from rdkit.Contrib.SA_Score import sascorer
except ImportError:  # pragma: no cover - depends on RDKit distribution
    sascorer = None


def molecular_descriptors(smiles: str) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid standardized SMILES")
    mw = float(Descriptors.MolWt(mol))
    logp = float(Crippen.MolLogP(mol))
    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    hbd = int(Lipinski.NumHDonors(mol))
    hba = int(Lipinski.NumHAcceptors(mol))
    rot = int(Lipinski.NumRotatableBonds(mol))
    rings = int(rdMolDescriptors.CalcNumRings(mol))
    aromatic_rings = int(rdMolDescriptors.CalcNumAromaticRings(mol))
    fraction_csp3 = float(rdMolDescriptors.CalcFractionCSP3(mol))
    formal_charge = int(Chem.GetFormalCharge(mol))
    heavy_atoms = int(mol.GetNumHeavyAtoms())
    qed = float(Descriptors.qed(mol))
    sa_score = float(sascorer.calculateScore(mol)) if sascorer is not None else float("nan")
    sa_desirability = max(0.0, min(1.0, (10.0 - sa_score) / 9.0)) if sa_score == sa_score else float("nan")
    lipinski_violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    return {
        "mw": mw,
        "logp": logp,
        "tpsa": tpsa,
        "hbd": hbd,
        "hba": hba,
        "rotatable_bonds": rot,
        "ring_count": rings,
        "aromatic_ring_count": aromatic_rings,
        "fraction_csp3": fraction_csp3,
        "formal_charge": formal_charge,
        "heavy_atom_count": heavy_atoms,
        "qed": qed,
        "sa_score": sa_score,
        "sa_desirability": sa_desirability,
        "lipinski_violations": int(lipinski_violations),
    }
