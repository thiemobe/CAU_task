# CAU_task

Single-layer linear regression that maps MNIST digits 1, 2, and 3 to the letters C, A, and U.

## Baseline

All regression scripts use every example labeled 1, 2, or 3 in `imagesTrain` by default. They pair each digit image with its corresponding fixed letter target. To use only the first N training examples of each digit, pass `--max-samples-per-digit N` (for example, `--max-samples-per-digit 100`). Evaluation uses the separate official `imagesTest` split and is unaffected by this training cap. For digits 1, 2, and 3, that split has 3,177 examples (1,135, 1,032, and 1,010 respectively). The MAT file stores one example per column: a leading constant bias feature followed by 576 grayscale pixels for a 24 x 24 image in MATLAB column-major order. The bias feature is discarded, so the model has no additive bias and no hidden units.

Digit images are resized to each requested input resolution. Letter targets are loaded from `Outputs/{c,a,u}_{size}.png`. The default sweep trains all six combinations of 64 x 64, 8 x 8, or 4 x 4 inputs and 64 x 64 or 16 x 16 outputs. Training and held-out pixel MSE are averaged over their respective examples and output pixels; weights have shape `(output_pixels, input_pixels)`. Full-dataset runs at high resolutions can require substantial time and memory; use the sample cap or start with 8 x 8 inputs and outputs.

Install the packages in `requirements.txt`, then run:

```powershell
python train_regression.py
```

To run just the full-resolution baseline, for example:

```powershell
python train_regression.py --input-sizes 8 --output-sizes 8 --max-samples-per-digit 100
```

## CAU Hardware Regression

`Mistari/code/oneLayer_CAU_Regression_1Device_IHP.m` uses the full `imagesTrain` examples for digits 1, 2, and 3 by default, with `imagesTest` held out for test MSE. Set `MAX_SAMPLES_PER_DIGIT` in the script to a positive integer to limit training examples per class. On each training sample above the MSE threshold, it selects one output/input coordinate uniformly; if that coordinate's gradient is zero, it skips device programming. The epoch log includes training MSE, test MSE, zero-gradient skips, and applied updates. It saves the first three held-out test reconstructions for each digit. The hardware path was not exercised by software validation.

## Memristor surrogate

The surrogate trainers use `sample_weight_from_lookup()` from `mdn_sigma.py` for one randomly selected weight coordinate per eligible digit example. Updates require per-sample pixel MSE above `--sample-mse-threshold` (default `0.2`) and a misclassified selected output pixel. The pixel gate thresholds prediction and target at `0.5`; for grayscale runners this is only a gate, while MSE and gradients still use continuous grayscale values. The selected coordinate then uses the squared-error gradient `(predicted_pixel - target_pixel) * input_pixel`. The weights are constrained by the surrogate to `[-1, 1]`. This is stochastic coordinate training, so its MSE is not expected to match the software pseudoinverse baseline.

The original grayscale-target version is `train_regression_surrogate.py`:

```powershell
python train_regression_surrogate.py --mode surrogate --input-sizes 8 --output-sizes 8 --steps 10 --sample-mse-threshold 0.2 --max-samples-per-digit 100
```

`train_regression_surrogate_binary.py` uses every training image labeled 1, 2, or 3, pairs it with the corresponding letter target, binarizes targets at `0.5`, and requires both gates to pass. The MAT file has 6,742 digit-1, 5,958 digit-2, and 6,131 digit-3 examples. Each epoch visits all 18,831 examples, so large epoch counts can take substantially longer than the three-example variants. It also saves thresholded reconstructions:

```powershell
python train_regression_surrogate_binary.py --mode surrogate --input-sizes 8 --output-sizes 8 --steps 10 --sample-mse-threshold 0.2 --max-samples-per-digit 100
```

`train_regression_surrogate_voltage_window.py` restricts lookup-row selection to inclusive `0.4`-`0.7 V` entries for both SET and RESET:

```powershell
python train_regression_surrogate_voltage_window.py --mode surrogate --input-sizes 8 --output-sizes 8 --steps 10 --sample-mse-threshold 0.2 --max-samples-per-digit 100
```

Each surrogate runner writes to its own folder under `Outputs/regression_results/`: `surrogate/`, `surrogate_binary/`, or `surrogate_voltage_0p4_0p7/`. Capped runs are saved in a `firstN_per_digit/` subfolder so they do not overwrite full-data results. `metrics.csv` records training and test example counts by digit, training pixel MSE, held-out `test_pixel_mse`, lookup-update attempts, and effective scalar-weight changes. Reconstructions are generated from the first three held-out test examples for each digit. `training_curve.csv` records training MSE, samples passing the MSE and update gates, attempts, and effective changes per epoch. The lookup CSV defaults to `model_run_all_voltages.csv` in the project root; use `--lookup-csv` to select another file.

Results are written to `Outputs/regression_results/`: `metrics.csv` contains the per-resolution MSE and pseudoinverse comparison, `weights/` contains the trained weight matrices, and `reconstructions/` contains the model outputs. The gradient update is equivalent to plain full-batch gradient descent from zero weights, while keeping its intermediate state small because the three input vectors span the learned weights.
