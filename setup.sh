#!/bin/bash
set -e

echo "=== SimpleFold Hackathon Setup ==="

# 1. Create venv
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 2. Install package + deps
echo "Installing simplefold..."
pip install --upgrade pip
pip install -e .
pip install redis fairscale tensorboard

# 3. Create required directories
mkdir -p artifacts/checkpoints artifacts/tensorboard artifacts/samples logs tmp

# 4. Quick sanity check: train 20 steps
echo ""
echo "=== Sanity check: 20 training steps ==="
python src/simplefold/train.py \
    experiment=train_local \
    data=competition \
    data.num_workers=4 \
    data.pin_memory=False \
    trainer.max_steps=20

echo ""
echo "=== Setup complete ==="
echo ""
echo "Activate the environment with: source venv/bin/activate"
echo ""
echo "To run a team's submission:"
echo "  python competition/compete.py --team-name TeamX --submission teamx.zip --max-steps 5000 --gpu 0"
echo ""
echo "To show leaderboard:"
echo "  python competition/leaderboard.py"
