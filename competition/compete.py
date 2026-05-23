"""
Competition runner. Runs a team's submission end-to-end.

Usage:
    python competition/compete.py \
        --team-name "AlphaFolders" \
        --submission path/to/submission.zip \
        --max-steps 5000 \
        --gpu 0
"""

import os
import sys
import json
import shutil
import zipfile
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

import yaml


def extract_submission(zip_path: str, team_name: str, runs_dir: Path) -> Path:
    team_dir = runs_dir / team_name
    if team_dir.exists():
        shutil.rmtree(team_dir)
    team_dir.mkdir(parents=True)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(team_dir / "submission")

    configs = list((team_dir / "submission").rglob("config.yaml"))
    if not configs:
        raise FileNotFoundError("No config.yaml found in submission zip")

    print(f"Extracted submission to {team_dir / 'submission'}")
    print(f"  Config: {configs[0]}")

    train_scripts = list((team_dir / "submission").rglob("train.py"))
    if train_scripts:
        print(f"  Custom train.py: {train_scripts[0]}")

    return team_dir


def parse_team_config(team_dir: Path) -> dict:
    config_path = list((team_dir / "submission").rglob("config.yaml"))[0]
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_architecture_name(cfg: dict) -> str:
    defaults = cfg.get("defaults", [])
    for d in defaults:
        if isinstance(d, dict) and "override /model/architecture" in d:
            return d["override /model/architecture"]
    return "foldingdit_100M"


def run_training(team_dir: Path, max_steps: int, gpu: int):
    submission_dir = team_dir / "submission"
    config_path = list(submission_dir.rglob("config.yaml"))[0]

    # If team provided custom train.py, copy it into src/simplefold/ so imports work
    custom_train = list(submission_dir.rglob("train.py"))
    train_script_path = Path("src/simplefold/train.py")
    backup_path = Path("src/simplefold/_train_backup.py")
    used_custom = False
    if custom_train:
        shutil.copy2(train_script_path, backup_path)
        shutil.copy2(custom_train[0], train_script_path)
        used_custom = True

    artifacts_dir = team_dir / "artifacts"
    for subdir in ["checkpoints", "tensorboard", "samples"]:
        (artifacts_dir / subdir).mkdir(parents=True, exist_ok=True)

    team_config_name = f"_team_{team_dir.name}"
    dest = Path("configs/experiment") / f"{team_config_name}.yaml"
    shutil.copy2(config_path, dest)

    ckpt_interval = max(1, max_steps // 2)
    cmd = [
        sys.executable, str(train_script_path),
        f"experiment={team_config_name}",
        f"trainer.max_steps={max_steps}",
        f"trainer.devices=1",
        f"trainer.val_check_interval={ckpt_interval}",
        f"paths.output_dir={artifacts_dir}",
        f"callbacks.model_checkpoint.dirpath={artifacts_dir}/checkpoints",
        f"callbacks.model_checkpoint.every_n_train_steps={ckpt_interval}",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print(f"\n{'='*60}")
    print(f"  TRAINING: {team_dir.name} (max {max_steps} steps, GPU {gpu})")
    print(f"{'='*60}")
    result = subprocess.run(cmd, env=env)

    dest.unlink(missing_ok=True)
    if used_custom:
        shutil.copy2(backup_path, train_script_path)
        backup_path.unlink(missing_ok=True)

    ckpt_dir = artifacts_dir / "checkpoints"
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        raise RuntimeError(f"Training produced no checkpoints for {team_dir.name}")
    print(f"  Checkpoints: {[c.name for c in ckpts]}")


def run_inference(team_dir: Path, test_fastas_dir: Path, gpu: int, arch_name: str):
    ckpt_dir = team_dir / "artifacts" / "checkpoints"
    last_ckpt = ckpt_dir / "last.ckpt"
    if not last_ckpt.exists():
        ckpts = sorted(ckpt_dir.glob("*.ckpt"))
        last_ckpt = ckpts[-1] if ckpts else None
    if not last_ckpt:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    pred_dir = team_dir / "predictions"
    device = f"cuda:{gpu}" if gpu >= 0 else "cpu"

    cmd = [
        sys.executable, "competition/run_inference.py",
        "--checkpoint", str(last_ckpt),
        "--test-fastas", str(test_fastas_dir),
        "--output-dir", str(pred_dir),
        "--architecture", arch_name,
        "--num-steps", "200",
        "--device", device,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(max(gpu, 0))

    print(f"\n{'='*60}")
    print(f"  INFERENCE: {team_dir.name}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, env=env)

    preds = list(pred_dir.glob("*.pdb")) + list(pred_dir.glob("*.cif"))
    print(f"  Generated {len(preds)} prediction files")


def score_predictions(team_dir: Path, test_ref_dir: Path) -> dict:
    sys.path.insert(0, str(Path(__file__).parent))
    from scoring import score_prediction

    pred_dir = team_dir / "predictions"
    results = {}

    ref_files = sorted(test_ref_dir.glob("*.npz"))
    for ref_file in ref_files:
        protein_id = ref_file.stem

        # Look for prediction (pdb or cif)
        pred_pdb = pred_dir / f"{protein_id}.pdb"
        pred_cif = pred_dir / f"{protein_id}.cif"
        pred_path = pred_pdb if pred_pdb.exists() else (pred_cif if pred_cif.exists() else None)

        if not pred_path:
            results[protein_id] = {"lddt": 0.0, "error": "no prediction generated"}
            continue

        # Extract ground truth PDB from the npz for comparison
        # The ref structures are in npz format, we need to convert
        # For scoring, we compare against the ground truth mmcif from test_fastas processing
        gt_cif_candidates = list((team_dir / "predictions" / "_cache").rglob(f"*{protein_id}*/*.cif")) + \
                           list((team_dir / "predictions" / "_cache").rglob(f"*{protein_id}*.cif"))

        # Actually, we need the ground truth structure separately
        # Use the ref npz to extract CA coords directly
        try:
            ref_data = np.load(str(ref_file))
            ref_atoms = ref_data["atoms"]
            # atoms array: each row is an atom with coords at specific indices
            # For CA-only comparison, we'll use the scoring module on the predicted file
            # and compare against reference coords extracted from npz
            result = score_prediction(str(pred_path), str(pred_path))  # placeholder
            results[protein_id] = result
        except Exception as e:
            results[protein_id] = {"lddt": 0.0, "error": str(e)}

    lddts = [r["lddt"] for r in results.values() if "error" not in r]
    mean_lddt = sum(lddts) / len(lddts) if lddts else 0.0

    return {"mean_lddt": mean_lddt, "per_protein": results}


def score_predictions_from_refs(team_dir: Path, test_ref_cif_dir: Path) -> dict:
    """Score predictions against ground truth CIF/PDB files."""
    sys.path.insert(0, str(Path(__file__).parent))
    from scoring import score_prediction

    pred_dir = team_dir / "predictions"
    results = {}

    ref_files = sorted(list(test_ref_cif_dir.glob("*.cif")) + list(test_ref_cif_dir.glob("*.pdb")))
    if not ref_files:
        print(f"  WARNING: No reference files found in {test_ref_cif_dir}")
        return {"mean_lddt": 0.0, "per_protein": {}}

    for ref_file in ref_files:
        protein_id = ref_file.stem

        pred_pdb = pred_dir / f"{protein_id}.pdb"
        pred_cif = pred_dir / f"{protein_id}.cif"
        pred_path = pred_pdb if pred_pdb.exists() else (pred_cif if pred_cif.exists() else None)

        if not pred_path:
            results[protein_id] = {"lddt": 0.0, "error": "no prediction"}
            continue

        results[protein_id] = score_prediction(str(pred_path), str(ref_file))

    lddts = [r["lddt"] for r in results.values() if "error" not in r]
    mean_lddt = sum(lddts) / len(lddts) if lddts else 0.0
    return {"mean_lddt": mean_lddt, "per_protein": results}


def update_leaderboard(team_name: str, scores: dict, leaderboard_path: Path) -> list:
    if leaderboard_path.exists():
        leaderboard = json.loads(leaderboard_path.read_text())
    else:
        leaderboard = []

    entry = {
        "rank": 0,
        "team": team_name,
        "mean_lddt": round(scores["mean_lddt"], 4),
        "num_scored": len([v for v in scores["per_protein"].values() if "error" not in v]),
        "num_failed": len([v for v in scores["per_protein"].values() if "error" in v]),
        "timestamp": datetime.now().isoformat(),
    }

    leaderboard = [e for e in leaderboard if e["team"] != team_name]
    leaderboard.append(entry)
    leaderboard.sort(key=lambda x: x["mean_lddt"], reverse=True)
    for i, e in enumerate(leaderboard):
        e["rank"] = i + 1

    leaderboard_path.write_text(json.dumps(leaderboard, indent=2))
    return leaderboard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team-name", required=True)
    parser.add_argument("--submission", required=True, help="Path to submission zip")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--test-ref-dir", type=str, default=None,
                        help="Dir with ground truth CIF/PDB files for scoring")
    args = parser.parse_args()

    comp_dir = Path(__file__).parent
    runs_dir = comp_dir / "runs"
    test_fastas_dir = comp_dir / "test_fastas"
    leaderboard_path = comp_dir / "leaderboard.json"

    if args.test_ref_dir:
        test_ref_dir = Path(args.test_ref_dir)
    else:
        test_ref_dir = comp_dir / "test_data" / "ground_truth_structures"

    # 1. Extract
    team_dir = extract_submission(args.submission, args.team_name, runs_dir)

    # Parse config to get architecture name
    cfg = parse_team_config(team_dir)
    arch_name = get_architecture_name(cfg)
    print(f"  Architecture: {arch_name}")

    # 2. Train
    if not args.skip_training:
        run_training(team_dir, args.max_steps, args.gpu)

    # 3. Infer
    if not args.skip_inference:
        run_inference(team_dir, test_fastas_dir, args.gpu, arch_name)

    # 4. Score
    scores = score_predictions_from_refs(team_dir, test_ref_dir)
    results_path = team_dir / "results.json"
    results_path.write_text(json.dumps(scores, indent=2))

    # 5. Leaderboard
    leaderboard = update_leaderboard(args.team_name, scores, leaderboard_path)

    print(f"\n{'='*60}")
    print(f"  RESULTS: {args.team_name}")
    print(f"  Mean lDDT: {scores['mean_lddt']:.4f}")
    print(f"{'='*60}")
    print(f"\n  Per-protein scores:")
    for pid, result in sorted(scores["per_protein"].items()):
        lddt = result.get("lddt", 0.0)
        err = result.get("error", "")
        status = f"  {err}" if err else ""
        print(f"    {pid}: {lddt:.4f}{status}")

    print(f"\n  LEADERBOARD:")
    for entry in leaderboard:
        marker = " <--" if entry["team"] == args.team_name else ""
        print(f"    {entry['rank']}. {entry['team']}: {entry['mean_lddt']:.4f}{marker}")


if __name__ == "__main__":
    main()
