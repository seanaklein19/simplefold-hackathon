import numpy as np
from pathlib import Path
from Bio.PDB import MMCIFParser, PDBParser


def extract_ca_coords(structure_path: str) -> np.ndarray:
    path = Path(structure_path)
    if path.suffix in ('.cif', '.mmcif'):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    structure = parser.get_structure('s', str(path))
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    coords.append(residue['CA'].get_vector().get_array())
        break
    return np.array(coords)


def compute_lddt(
    pred_coords: np.ndarray,
    ref_coords: np.ndarray,
    cutoff: float = 15.0,
    thresholds: tuple = (0.5, 1.0, 2.0, 4.0),
) -> float:
    assert pred_coords.shape == ref_coords.shape
    n = len(ref_coords)
    if n < 2:
        return 0.0

    ref_dists = np.sqrt(((ref_coords[:, None] - ref_coords[None, :]) ** 2).sum(-1))
    pred_dists = np.sqrt(((pred_coords[:, None] - pred_coords[None, :]) ** 2).sum(-1))

    mask = (ref_dists < cutoff) & (np.eye(n) == 0)
    if mask.sum() == 0:
        return 0.0

    diff = np.abs(ref_dists - pred_dists)
    scores = [(diff[mask] < t).mean() for t in thresholds]
    return float(np.mean(scores))


def score_prediction(pred_path: str, ref_path: str) -> dict:
    try:
        pred_ca = extract_ca_coords(pred_path)
        ref_ca = extract_ca_coords(ref_path)
    except Exception as e:
        return {"lddt": 0.0, "num_residues": 0, "error": str(e)}

    min_len = min(len(pred_ca), len(ref_ca))
    if min_len < 2:
        return {"lddt": 0.0, "num_residues": min_len, "error": "too few CA atoms"}

    pred_ca = pred_ca[:min_len]
    ref_ca = ref_ca[:min_len]
    lddt = compute_lddt(pred_ca, ref_ca)
    return {"lddt": lddt, "num_residues": min_len}
