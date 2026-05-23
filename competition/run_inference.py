"""
Run inference on test proteins using a training checkpoint.
Uses pre-processed test structures directly (no downloads needed).

Usage:
    python competition/run_inference.py \
        --checkpoint competition/runs/team/artifacts/checkpoints/last.ckpt \
        --test-dir competition/test_data \
        --output-dir competition/runs/team/predictions/ \
        --architecture foldingdit_100M \
        --num-steps 200 \
        --device cuda:0
"""

import os
import sys
import copy
import json
import torch
import hydra
import argparse
import omegaconf
import numpy as np
from pathlib import Path
from importlib import resources
import lightning.pytorch as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "simplefold"))

from model.flow import LinearPath
from model.torch.sampler import EMSampler
from processor.protein_processor import ProteinDataProcessor
from utils.datamodule_utils import process_one_inference_structure, collate
from utils.esm_utils import _af2_to_esm, esm_registry
from utils.boltz_utils import process_structure as boltz_process_structure, save_structure
from boltz_data_pipeline.feature.featurizer import BoltzFeaturizer
from boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer
from boltz_data_pipeline.types import Structure, Record


def load_model_from_training_ckpt(ckpt_path: str, arch_config_name: str, device: torch.device):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    config_files = resources.files('simplefold.configs')
    cfg_path = str(config_files / "model" / "architecture" / f"{arch_config_name}.yaml")
    model_config = omegaconf.OmegaConf.load(cfg_path)
    model = hydra.utils.instantiate(model_config)

    state_dict = checkpoint.get("state_dict", checkpoint)

    ema_prefix = "model_ema.module.architecture."
    ema_keys = {k: k[len(ema_prefix):] for k in state_dict if k.startswith(ema_prefix)}

    if ema_keys:
        new_state = {new_k: state_dict[old_k] for old_k, new_k in ema_keys.items()}
        model.load_state_dict(new_state, strict=False)
        print(f"Loaded EMA weights ({len(ema_keys)} params)")
    else:
        arch_prefix = "architecture."
        arch_keys = {k: k[len(arch_prefix):] for k in state_dict if k.startswith(arch_prefix)}
        if arch_keys:
            new_state = {new_k: state_dict[old_k] for old_k, new_k in arch_keys.items()}
            model.load_state_dict(new_state, strict=False)
            print(f"Loaded architecture weights ({len(arch_keys)} params)")
        else:
            model.load_state_dict(state_dict, strict=False)
            print("Loaded weights directly")

    model = model.to(device)
    model.eval()
    return model


def run_inference(args):
    device = torch.device(args.device)
    pl.seed_everything(42)

    print(f"Loading model from {args.checkpoint}")
    model = load_model_from_training_ckpt(args.checkpoint, args.architecture, device)

    print(f"Loading ESM model: esm2_8M")
    esm_model, esm_dict = esm_registry["esm2_8M"]()
    esm_model = esm_model.to(device)
    esm_model.eval()
    af2_to_esm = _af2_to_esm(esm_dict).to(device)

    tokenizer = BoltzTokenizer()
    featurizer = BoltzFeaturizer()
    processor = ProteinDataProcessor(
        device=device, scale=16.0, ref_scale=5.0,
        multiplicity=1, inference_multiplicity=1, backend="torch",
    )
    flow = LinearPath()
    sampler = EMSampler(
        num_timesteps=args.num_steps, t_start=1e-4,
        tau=args.tau, log_timesteps=True, w_cutoff=0.99,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_dir = Path(args.test_dir)
    struct_dir = test_dir / "processed" / "structures"
    record_dir = test_dir / "processed" / "records"

    struct_files = sorted(struct_dir.glob("*.npz"))
    print(f"Running inference on {len(struct_files)} proteins")

    for struct_file in struct_files:
        protein_id = struct_file.stem
        record_file = record_dir / f"{protein_id}.json"
        print(f"  Processing {protein_id}...")

        if not record_file.exists():
            print(f"    Skipping {protein_id}: no record file")
            continue

        try:
            batch, structure, record = process_one_inference_structure(
                str(struct_file), str(record_file),
                tokenizer, featurizer, processor,
                esm_model=esm_model, esm_dict=esm_dict,
                af2_to_esm=af2_to_esm,
            )

            batch = processor.batch_to_device(batch)
            noise = torch.randn_like(batch['coords']).to(device)

            with torch.no_grad():
                out_dict = sampler.sample(model.forward, flow, noise, batch)
                out_dict = processor.postprocess(out_dict, batch)

            sampled_coord = out_dict['denoised_coords'].detach()
            pad_mask = batch['atom_pad_mask']

            sampled_structure = copy.deepcopy(structure)
            sampled_structure = boltz_process_structure(
                sampled_structure, sampled_coord[0], pad_mask[0], record
            )
            save_structure(sampled_structure, output_dir, protein_id, output_format="pdb")
            save_structure(sampled_structure, output_dir, protein_id, output_format="mmcif")
            print(f"    Saved prediction for {protein_id}")

        except Exception as e:
            print(f"    Failed on {protein_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nPredictions saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--architecture", default="foldingdit_100M")
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
