"""
One-time setup: downloads PDB structures, processes them, splits train/test.

Prerequisites:
    conda activate simplefold
    redis-server --dbfilename ccd.rdb --dir data/ --port 7777 --daemonize yes

Usage:
    python competition/prepare_competition.py [--num-train 30] [--num-test 10]
"""

import os
import sys
import json
import pickle
import random
import shutil
import argparse
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "simplefold"))

from process_mmcif import fetch, process_structure, finalize, Resource, PDB
from process_structure import tokenize_structure, finalize as finalize_tokens
from boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer
from boltz_data_pipeline.types import Manifest
from boltz_data_pipeline.filter.static.filter import StaticFilter
from boltz_data_pipeline.filter.static.ligand import ExcludedLigands
from boltz_data_pipeline.filter.static.polymer import (
    MinimumLengthFilter, UnknownFilter, ConsecutiveCA, ClashingChainsFilter
)

import rdkit
import numpy as np


def download_cifs(pdb_ids: list[str], out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for pdb_id in pdb_ids:
        dest = out_dir / f"{pdb_id}.cif"
        if dest.exists():
            downloaded.append(pdb_id)
            continue
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        result = subprocess.run(
            ["curl", "-sL", "-o", str(dest), url],
            capture_output=True, timeout=30,
        )
        if result.returncode == 0 and dest.stat().st_size > 100:
            downloaded.append(pdb_id)
        else:
            dest.unlink(missing_ok=True)
            print(f"  Failed to download {pdb_id}")
    return downloaded


def process_mmcifs(cif_dir: Path, out_dir: Path, redis_port: int) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "structures").mkdir(exist_ok=True)
    (out_dir / "records").mkdir(exist_ok=True)

    pickle_option = rdkit.Chem.PropertyPickleOptions.AllProps
    rdkit.Chem.SetDefaultPickleProperties(pickle_option)

    resource = Resource(host="localhost", port=redis_port)
    filters = [
        ExcludedLigands(),
        MinimumLengthFilter(min_len=4, max_len=5000),
        UnknownFilter(),
        ConsecutiveCA(max_dist=10.0),
        ClashingChainsFilter(freq=0.3, dist=1.7),
    ]

    data = fetch(cif_dir)
    succeeded = []
    for item in data:
        try:
            process_structure(item, resource, out_dir, filters, clusters={})
            struct_path = out_dir / "structures" / f"{item.id}.npz"
            if struct_path.exists():
                succeeded.append(item.id)
        except Exception as e:
            print(f"  Failed to process {item.id}: {e}")

    finalize(out_dir)
    return succeeded


def tokenize_structures(target_dir: Path, token_dir: Path) -> list[str]:
    token_dir.mkdir(parents=True, exist_ok=True)
    (token_dir / "tokens").mkdir(exist_ok=True)
    (token_dir / "records").mkdir(exist_ok=True)

    manifest = Manifest.load(target_dir / "manifest.json")
    tokenizer = BoltzTokenizer()
    succeeded = []
    for record in manifest.records:
        ok = tokenize_structure(
            record, tokenizer, target_dir,
            str(token_dir / "tokens"), token_dir / "records",
        )
        if ok:
            succeeded.append(record.id)

    finalize_tokens(token_dir)
    return succeeded


def extract_fastas(target_dir: Path, token_dir: Path, fasta_dir: Path, protein_ids: list[str]):
    fasta_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "simplefold"))
    from utils.datamodule_utils import extract_sequence_from_tokens
    from boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer
    from boltz_data_pipeline.types import Structure, Input, Connection

    tokenizer = BoltzTokenizer()
    for pid in protein_ids:
        try:
            structure = np.load(target_dir / "structures" / f"{pid}.npz")
            structure = Structure(
                atoms=structure["atoms"], bonds=structure["bonds"],
                residues=structure["residues"], chains=structure["chains"],
                connections=structure["connections"].astype(Connection),
                interfaces=structure["interfaces"], mask=structure["mask"],
            )
            input_data = Input(structure, {})
            tokenized = tokenizer.tokenize(input_data)
            seq = extract_sequence_from_tokens(tokenized)
            seq_clean = seq.replace(":", "\n>chain\n")
            fasta_path = fasta_dir / f"{pid}.fasta"
            with open(fasta_path, "w") as f:
                f.write(f">{pid}\n{seq.split(':')[0]}\n")
        except Exception as e:
            print(f"  Failed to extract FASTA for {pid}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-train", type=int, default=30)
    parser.add_argument("--num-test", type=int, default=10)
    parser.add_argument("--redis-port", type=int, default=7777)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    comp_dir = Path(__file__).parent
    pdb_ids_file = comp_dir / "pdb_ids.txt"
    pdb_ids = [line.strip() for line in pdb_ids_file.read_text().splitlines() if line.strip()]

    print(f"Loaded {len(pdb_ids)} PDB IDs")

    # Step 1: Download
    print("\n=== Downloading CIF files ===")
    cif_dir = comp_dir / "raw_cifs"
    downloaded = download_cifs(pdb_ids, cif_dir)
    print(f"Downloaded {len(downloaded)}/{len(pdb_ids)} structures")

    # Step 2: Process mmcif
    print("\n=== Processing MMCIF files ===")
    all_processed_dir = comp_dir / "_all_processed"
    succeeded = process_mmcifs(cif_dir, all_processed_dir, args.redis_port)
    print(f"Processed {len(succeeded)}/{len(downloaded)} structures")

    # Step 3: Tokenize
    print("\n=== Tokenizing structures ===")
    all_tokenized_dir = comp_dir / "_all_tokenized"
    tokenized = tokenize_structures(all_processed_dir, all_tokenized_dir)
    print(f"Tokenized {len(tokenized)}/{len(succeeded)} structures")

    # Step 4: Split train/test
    total_needed = args.num_train + args.num_test
    if len(tokenized) < total_needed:
        print(f"\nWARNING: Only {len(tokenized)} proteins available, "
              f"need {total_needed}. Adjusting split.")
        args.num_test = max(5, len(tokenized) // 4)
        args.num_train = len(tokenized) - args.num_test

    random.seed(args.seed)
    random.shuffle(tokenized)
    test_ids = sorted(tokenized[:args.num_test])
    train_ids = sorted(tokenized[args.num_test:args.num_test + args.num_train])

    print(f"\nSplit: {len(train_ids)} train, {len(test_ids)} test")
    print(f"Train: {train_ids}")
    print(f"Test:  {test_ids}")

    # Step 5: Copy to train/test dirs
    for split_name, split_ids in [("train_data", train_ids), ("test_data", test_ids)]:
        split_dir = comp_dir / split_name
        proc_dir = split_dir / "processed"
        (proc_dir / "structures").mkdir(parents=True, exist_ok=True)
        (proc_dir / "records").mkdir(parents=True, exist_ok=True)
        tok_dir = split_dir / "tokenized"
        (tok_dir / "tokens").mkdir(parents=True, exist_ok=True)
        (tok_dir / "records").mkdir(parents=True, exist_ok=True)

        records = []
        for pid in split_ids:
            # Copy structure
            src = all_processed_dir / "structures" / f"{pid}.npz"
            if src.exists():
                shutil.copy2(src, proc_dir / "structures" / f"{pid}.npz")
            # Copy processed record
            src = all_processed_dir / "records" / f"{pid}.json"
            if src.exists():
                shutil.copy2(src, proc_dir / "records" / f"{pid}.json")
                records.append(json.load(open(src)))
            # Copy token
            src = all_tokenized_dir / "tokens" / f"{pid}.pkl"
            if src.exists():
                shutil.copy2(src, tok_dir / "tokens" / f"{pid}.pkl")
            # Copy token record
            src = all_tokenized_dir / "records" / f"{pid}.json"
            if src.exists():
                shutil.copy2(src, tok_dir / "records" / f"{pid}.json")

        # Write manifests
        with open(proc_dir / "manifest.json", "w") as f:
            json.dump(records, f)
        # Token manifest - reload token records
        token_records = []
        for pid in split_ids:
            rec_path = tok_dir / "records" / f"{pid}.json"
            if rec_path.exists():
                token_records.append(json.load(open(rec_path)))
        with open(tok_dir / "manifest.json", "w") as f:
            json.dump(token_records, f)

    # Step 6: Extract FASTA files for test proteins
    print("\n=== Extracting test FASTA files ===")
    extract_fastas(
        comp_dir / "test_data" / "processed",
        comp_dir / "test_data" / "tokenized",
        comp_dir / "test_fastas",
        test_ids,
    )

    # Step 7: Create symmetry.pkl
    with open(comp_dir / "symmetry.pkl", "wb") as f:
        pickle.dump({}, f)

    # Step 8: Save split info
    split_info = {"train": train_ids, "test": test_ids, "seed": args.seed}
    with open(comp_dir / "split.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # Cleanup temp dirs
    shutil.rmtree(all_processed_dir, ignore_errors=True)
    shutil.rmtree(all_tokenized_dir, ignore_errors=True)

    print(f"\n=== Done ===")
    print(f"Train data: {comp_dir / 'train_data'}")
    print(f"Test data:  {comp_dir / 'test_data'}")
    print(f"Test FASTAs: {comp_dir / 'test_fastas'}")
    print(f"Split info: {comp_dir / 'split.json'}")


if __name__ == "__main__":
    main()
