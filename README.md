# SimpleFold Hackathon

Train the best protein structure prediction model in 5000 steps.

Built on [SimpleFold](https://github.com/apple/ml-simplefold) (Apple, 2025) - a flow-matching protein folding model.

## How It Works

Teams modify a config file (and optionally `train.py`) to tune how SimpleFold trains. Everyone gets the same 30 training proteins (ubiquitin-like fold family), the same compute budget (5000 steps), and the same frozen ESM encoder. Submissions are evaluated on 10 hidden test proteins by mean lDDT score.

```
Protein Sequence → [ESM encoder (frozen)] → [FoldingDiT (you train this)] → 3D Structure
```

**What you can change:**
- Model architecture (100M, 360M, 700M)
- Learning rate, warmup schedule, weight decay
- Loss function weights (MSE vs lDDT balance)
- Multiplicity (noised copies per protein per step)
- Gradient accumulation, EMA decay, gradient clipping
- Code changes to `train.py`

## Setup

```bash
conda create -n simplefold python=3.10 -y
conda activate simplefold
pip install -e .
pip install redis fairscale tensorboard
```

## For Participants

Your starter kit is in `competition/starter_kit/`:
- `config.yaml` - modify this to tune your model
- `README.md` - rules and tips

Test locally (CPU, fast):
```bash
python src/simplefold/train.py experiment=train_local data=competition data.num_workers=0 data.pin_memory=False trainer.max_steps=50
```

Submit:
```bash
zip submission.zip config.yaml train.py  # train.py is optional
```

## For Organizers

### First-time setup (GPU server)

```bash
git clone https://github.com/seanaklein19/simplefold-hackathon.git
cd simplefold-hackathon
conda create -n simplefold python=3.10 -y && conda activate simplefold
pip install -e . && pip install redis fairscale tensorboard
```

### Run a team's submission

```bash
python competition/compete.py \
    --team-name "TeamName" \
    --submission path/to/submission.zip \
    --max-steps 5000 \
    --gpu 0
```

### Show leaderboard

```bash
python competition/leaderboard.py                          # terminal
python competition/leaderboard.py --export html             # HTML (auto-refreshes, put on projector)
python competition/leaderboard.py --detailed                # per-protein breakdown
```

## Dataset

30 train / 10 test proteins from the **ubiquitin-like fold family** (CATH 3.10.20). All single-chain, 56-140 residues, X-ray crystallography. Same fold topology ensures training transfers to the test set.

## Scoring

Mean lDDT (Local Distance Difference Test) on the 10 held-out test proteins. lDDT measures how well local inter-residue distances are preserved compared to the true structure. Range: 0 (random) to 1 (perfect).
