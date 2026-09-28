# CAU_task

Single-layer linear regression that maps MNIST digits 1, 2, and 3 to the letters C, A, and U.

## Baseline

The script uses the first training example for each digit, so it matches the initial three-pair experiment. It does not measure generalization to other handwriting styles. The MAT file stores one example per column: a leading constant bias feature followed by 576 grayscale pixels for a 24 x 24 image in MATLAB column-major order. The bias feature is discarded, so the model has no additive bias and no hidden units.

Digit images are resized to each requested input resolution. Letter targets are loaded from `Outputs/{c,a,u}_{size}.png`. The default sweep trains all six combinations of 64 x 64, 8 x 8, or 4 x 4 inputs and 64 x 64 or 16 x 16 outputs. Pixel MSE is averaged over the three examples and all output pixels; weights have shape `(output_pixels, input_pixels)`.

Install the packages in `requirements.txt`, then run:

```powershell
python train_regression.py
```

To run just the full-resolution baseline, for example:

```powershell
python train_regression.py --input-sizes 64 --output-sizes 64
```

Results are written to `Outputs/regression_results/`: `metrics.csv` contains the per-resolution MSE and pseudoinverse comparison, `weights/` contains the trained weight matrices, and `reconstructions/` contains the model outputs. The gradient update is equivalent to plain full-batch gradient descent from zero weights, while keeping its intermediate state small because the three input vectors span the learned weights.
