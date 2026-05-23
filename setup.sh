#!/bin/bash
set -e

echo "=== SimpleFold Hackathon Setup ==="

# 1. Create venv
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# 2. Install package and deps (this will install a default torch)
echo "Installing simplefold..."
pip install -e .
pip install redis fairscale tensorboard

# 3. NOW reinstall torch with correct CUDA version (overrides the default)
if command -v nvidia-smi &> /dev/null; then
    CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    CUDA_MAJOR=$(echo $CUDA_VER | cut -d. -f1)
    CUDA_MINOR=$(echo $CUDA_VER | cut -d. -f2)
    echo "Detected CUDA ${CUDA_VER}, reinstalling PyTorch with GPU support..."

    INSTALLED=false
    for tag in "cu${CUDA_MAJOR}${CUDA_MINOR}" "cu${CUDA_MAJOR}$((CUDA_MINOR-1))" "cu${CUDA_MAJOR}$((CUDA_MINOR-2))" "cu124" "cu121"; do
        echo "Trying PyTorch for ${tag}..."
        if pip install torch torchvision --index-url "https://download.pytorch.org/whl/${tag}" --force-reinstall 2>/dev/null; then
            echo "Installed PyTorch for ${tag}"
            INSTALLED=true
            break
        fi
    done

    if [ "$INSTALLED" = false ]; then
        echo "WARNING: Could not install CUDA-compatible PyTorch. Falling back to default."
    fi

    # Verify
    python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, available: {torch.cuda.is_available()}')"
fi

# 4. Create required directories
mkdir -p artifacts/checkpoints artifacts/tensorboard artifacts/samples logs tmp

# 5. Quick sanity check: train 20 steps
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
