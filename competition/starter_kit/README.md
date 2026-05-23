# SimpleFold Hackathon

## Goal

Train the best protein structure prediction model in 5000 steps.
Scored by **mean lDDT** on held-out test proteins (higher = better, max 1.0).

## How SimpleFold Works

```
Protein Sequence  "MKTLLILAVL..."
       |
   [ESM encoder]     <- FROZEN, same for everyone (esm2_8M)
       |
   [FoldingDiT]      <- THIS is what you're training
       |              A transformer that learns to denoise
       |              3D coordinates via flow matching
       |
   3D Structure      (predicted atom positions)
```

**Training:** Take a real protein structure, add noise, train the model to undo the noise.
**Inference:** Start from pure noise, iteratively denoise to predict a structure.

## What You Can Change

**Config parameters (config.yaml):**
- Architecture: `foldingdit_100M`, `foldingdit_360M`, `foldingdit_700M` (bigger = more capacity but slower)
- Learning rate, warmup steps, weight decay
- `smooth_lddt_loss_weight` - balance between MSE and lDDT losses
- `multiplicity` - noised copies per protein per step (more = better gradients, slower)
- `accumulate_grad_batches` - effective batch size multiplier
- `ema_decay` - exponential moving average of weights
- `clip_grad_norm_val` - gradient clipping threshold
- Max tokens / crop size in data config

**Code (train.py):**
- Custom loss functions
- Modified training loop
- Flow schedule changes
- Data augmentation strategies

## What You Cannot Change

- ESM model (fixed at `esm2_8M`)
- Training data (provided, same for all teams)
- Step budget (5000 steps, enforced by organizers)

## How to Submit

1. Modify `config.yaml` and optionally `train.py`
2. Zip them: `zip submission.zip config.yaml train.py`
3. Submit the zip to organizers

## Local Testing

Train locally to iterate fast (CPU, fewer steps):

```bash
conda activate simplefold
python src/simplefold/train.py experiment=train_local
```

This runs ~100 steps on CPU to verify your config works.
Check the loss values to see if your changes help.

## Tips

- **Start with learning rate.** Try 3e-4, 5e-4, 1e-3. This often matters most.
- **Multiplicity is free signal.** More noised copies per step = better gradient estimates.
  But each copy costs memory, so there's a limit per GPU.
- **Loss weight matters.** The MSE loss trains global shape, lDDT loss trains local contacts.
  Try `smooth_lddt_loss_weight: 2.0` or `0.5`.
- **Bigger model != better.** With only 5000 steps, a 100M model may learn more than a 700M model
  that barely starts training.
- **Warmup is important.** Too little warmup → training diverges. Too much → wasted steps.
- **Look at the loss curves.** If MSE loss is still dropping at step 5000, you might benefit from
  higher learning rate or more aggressive warmup.
