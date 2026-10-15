import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

from mdn_sigma import load_lookup_table, sample_weight_from_lookup

# python train_regression_surrogate_binary.py --mode surrogate --input-sizes 8 --output-sizes 8 --steps 10 --sample-mse-threshold 0.2 --max-samples-per-digit 100

ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "Inputs" / "MNISTdataset0123.mat"
OUTPUTS_PATH = ROOT / "Outputs"
RESULTS_PATH = OUTPUTS_PATH / "regression_results"
DEFAULT_LOOKUP_PATH = ROOT / "model_run_all_voltages.csv"
DIGIT_TO_LETTER = {1: "c", 2: "a", 3: "u"}


def resize_grayscale(image: np.ndarray, size: int) -> np.ndarray:
    pixels = np.rint(np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    resized = Image.fromarray(pixels).resize(
        (size, size), Image.Resampling.BILINEAR
    )
    return np.asarray(resized, dtype=np.float64) / 255.0


def load_digit_examples() -> tuple[np.ndarray, list[int]]:
    data = loadmat(INPUT_PATH)
    if "imagesTrain" not in data or "labelsTrain" not in data:
        raise ValueError("MAT file must contain imagesTrain and labelsTrain.")

    images = np.asarray(data["imagesTrain"], dtype=np.float64)
    labels = np.asarray(data["labelsTrain"]).ravel()
    if images.ndim != 2 or images.shape[0] < 2:
        raise ValueError("imagesTrain must have features in rows and examples in columns.")
    if not np.allclose(images[0], 1.0):
        raise ValueError("Expected the first imagesTrain row to be the bias feature.")

    pixel_count = images.shape[0] - 1
    image_side = int(np.sqrt(pixel_count))
    if image_side * image_side != pixel_count:
        raise ValueError(f"Expected a square image after removing bias; got {pixel_count} pixels.")

    examples = []
    sample_indices = []
    for digit in DIGIT_TO_LETTER:
        matching_indices = np.flatnonzero(labels == digit)
        if matching_indices.size == 0:
            raise ValueError(f"No training example found for digit {digit}.")
        sample_index = int(matching_indices[0])
        image = images[1:, sample_index].reshape((image_side, image_side), order="F")
        examples.append(image)
        sample_indices.append(sample_index)

    return np.stack(examples), sample_indices


def load_all_digit_examples(
    max_samples_per_digit: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    if max_samples_per_digit is not None and max_samples_per_digit <= 0:
        raise ValueError("max_samples_per_digit must be a positive integer or None.")

    data = loadmat(INPUT_PATH)
    if "imagesTrain" not in data or "labelsTrain" not in data:
        raise ValueError("MAT file must contain imagesTrain and labelsTrain.")

    images = np.asarray(data["imagesTrain"], dtype=np.float64)
    labels = np.asarray(data["labelsTrain"]).ravel()
    if images.ndim != 2 or images.shape[0] < 2:
        raise ValueError("imagesTrain must have features in rows and examples in columns.")
    if not np.allclose(images[0], 1.0):
        raise ValueError("Expected the first imagesTrain row to be the bias feature.")

    pixel_count = images.shape[0] - 1
    image_side = int(np.sqrt(pixel_count))
    if image_side * image_side != pixel_count:
        raise ValueError(f"Expected a square image after removing bias; got {pixel_count} pixels.")

    selected_indices = []
    for digit in DIGIT_TO_LETTER:
        digit_indices = np.flatnonzero(labels == digit)
        if max_samples_per_digit is not None:
            digit_indices = digit_indices[:max_samples_per_digit]
        selected_indices.extend(digit_indices.tolist())
    sample_indices = sorted(selected_indices)
    if not sample_indices:
        raise ValueError("No training examples found for digits 1, 2, or 3.")
    sample_digit_labels = labels[sample_indices].astype(int)
    examples = [
        images[1:, sample_index].reshape((image_side, image_side), order="F")
        for sample_index in sample_indices
    ]
    return np.stack(examples), sample_digit_labels, sample_indices


def load_test_digit_examples() -> tuple[np.ndarray, np.ndarray, list[int]]:
    data = loadmat(INPUT_PATH)
    if "imagesTest" not in data or "labelsTest" not in data:
        raise ValueError("MAT file must contain imagesTest and labelsTest for evaluation.")

    images = np.asarray(data["imagesTest"], dtype=np.float64)
    labels = np.asarray(data["labelsTest"]).ravel()
    if images.ndim != 2 or images.shape[0] < 2:
        raise ValueError("imagesTest must have features in rows and examples in columns.")
    if not np.allclose(images[0], 1.0):
        raise ValueError("Expected the first imagesTest row to be the bias feature.")

    pixel_count = images.shape[0] - 1
    image_side = int(np.sqrt(pixel_count))
    if image_side * image_side != pixel_count:
        raise ValueError(f"Expected a square image after removing bias; got {pixel_count} pixels.")

    sample_indices = np.flatnonzero(np.isin(labels, list(DIGIT_TO_LETTER))).tolist()
    if not sample_indices:
        raise ValueError("No test examples found for digits 1, 2, or 3.")
    sample_digit_labels = labels[sample_indices].astype(int)
    examples = [
        images[1:, sample_index].reshape((image_side, image_side), order="F")
        for sample_index in sample_indices
    ]
    return np.stack(examples), sample_digit_labels, sample_indices


def load_letter_target(
    letter: str, size: int, binarization_threshold=None
) -> np.ndarray:
    target_path = OUTPUTS_PATH / f"{letter}_{size}.png"
    if not target_path.is_file():
        raise FileNotFoundError(f"Missing target image: {target_path}")
    with Image.open(target_path) as image:
        grayscale = image.convert("L").resize(
            (size, size), Image.Resampling.BILINEAR
        )
        pixels = np.asarray(grayscale, dtype=np.float64) / 255.0
    if binarization_threshold is not None:
        return (pixels >= binarization_threshold).astype(np.float64)
    return pixels


def train_linear_regression(
    inputs: np.ndarray, targets: np.ndarray, steps: int
) -> tuple[np.ndarray, float]:
    sample_count = inputs.shape[0]
    gram = inputs @ inputs.T
    largest_eigenvalue = np.linalg.eigvalsh(gram / sample_count).max()
    if largest_eigenvalue <= 0:
        raise ValueError("Input examples have no nonzero pixel values.")
    learning_rate = 1.0 / largest_eigenvalue

    # From zero initialization, weights stay in the span of the three inputs.
    coefficients = np.zeros((targets.shape[1], sample_count), dtype=np.float64)
    for _ in range(steps):
        errors = gram @ coefficients.T - targets
        coefficients -= learning_rate * errors.T / sample_count

    weights = coefficients @ inputs
    return weights, learning_rate


def train_surrogate_regression(
    inputs: np.ndarray,
    targets: np.ndarray,
    epochs: int,
    learning_rate: float,
    df_set,
    df_reset,
    seed: int,
    sample_mse_threshold: float = 0.1,
    pulse_voltage_range=None,
    update_gate: str = "mse",
    prediction_threshold: float = 0.5,
) -> tuple[np.ndarray, list[float], int, int, list[int], list[int], list[int], list[int]]:
    if update_gate not in {"mse", "misclassification"}:
        raise ValueError("update_gate must be 'mse' or 'misclassification'.")

    sample_count, input_features = inputs.shape
    output_features = targets.shape[1]
    weights = np.zeros((output_features, input_features), dtype=np.float64)
    predictions = np.zeros_like(targets)
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    mse_history = []
    samples_over_mse_threshold_history = []
    samples_passing_gate_history = []
    lookup_attempts_per_epoch = []
    effective_updates_per_epoch = []
    lookup_attempt_count = 0
    effective_update_count = 0

    for _ in range(epochs):
        samples_over_mse_threshold = 0
        samples_passing_gate = 0
        lookup_attempts = 0
        effective_updates = 0
        for sample_index in rng.permutation(sample_count):
            sample_mse = float(
                np.mean((predictions[sample_index] - targets[sample_index]) ** 2)
            )
            if sample_mse <= sample_mse_threshold:
                continue
            samples_over_mse_threshold += 1

            if update_gate == "mse":
                output_index = int(rng.integers(output_features))
            else:
                output_index = int(rng.integers(output_features))
                predicted_bit = predictions[sample_index, output_index] >= prediction_threshold
                target_bit = targets[sample_index, output_index] >= prediction_threshold
                if predicted_bit == target_bit:
                    continue
            samples_passing_gate += 1

            input_index = int(rng.integers(input_features))
            input_value = inputs[sample_index, input_index]
            gradient = (
                predictions[sample_index, output_index]
                - targets[sample_index, output_index]
            ) * input_value
            if gradient == 0.0:
                continue

            current_weight = weights[output_index, input_index]
            updated_weight, _, _ = sample_weight_from_lookup(
                w_current=current_weight,
                grad=learning_rate * gradient,
                df_set=df_set,
                df_reset=df_reset,
                voltage_range=pulse_voltage_range,
            )
            lookup_attempt_count += 1
            lookup_attempts += 1
            if not np.isclose(updated_weight, current_weight, rtol=0.0, atol=1e-12):
                effective_update_count += 1
                effective_updates += 1
            weights[output_index, input_index] = updated_weight
            predictions[:, output_index] += (
                updated_weight - current_weight
            ) * inputs[:, input_index]

        mse_history.append(float(np.mean((predictions - targets) ** 2)))
        samples_over_mse_threshold_history.append(samples_over_mse_threshold)
        samples_passing_gate_history.append(samples_passing_gate)
        lookup_attempts_per_epoch.append(lookup_attempts)
        effective_updates_per_epoch.append(effective_updates)

    return (
        weights,
        mse_history,
        lookup_attempt_count,
        effective_update_count,
        samples_over_mse_threshold_history,
        samples_passing_gate_history,
        lookup_attempts_per_epoch,
        effective_updates_per_epoch,
    )


def save_grayscale_image(
    path: Path, pixels: np.ndarray, size: int, binarization_threshold=None
) -> None:
    pixels = np.clip(pixels.reshape(size, size), 0.0, 1.0)
    if binarization_threshold is not None:
        pixels = (pixels >= binarization_threshold).astype(np.float64)
    image = np.rint(pixels * 255)
    Image.fromarray(image.astype(np.uint8)).save(path)


def run_experiments(
    input_sizes: list[int],
    output_sizes: list[int],
    steps: int,
    mode: str,
    lookup_csv: Path,
    learning_rate: float,
    seed: int,
    sample_mse_threshold: float = 0.2,
    result_variant: str = "surrogate",
    target_binarization_threshold=None,
    pulse_voltage_range=None,
    update_gate: str = "mse",
    use_all_digit_examples: bool = True,
    max_samples_per_digit: int | None = None,
) -> None:
    if update_gate not in {"mse", "misclassification"}:
        raise ValueError("update_gate must be 'mse' or 'misclassification'.")
    if update_gate == "misclassification" and mode != "surrogate":
        raise ValueError("Misclassification gating is available only in surrogate mode.")
    if target_binarization_threshold is not None and not (
        np.isfinite(target_binarization_threshold)
        and 0.0 <= target_binarization_threshold <= 1.0
    ):
        raise ValueError("Target binarization threshold must be between 0 and 1.")
    if pulse_voltage_range is not None and mode != "surrogate":
        raise ValueError("Pulse voltage filtering is available only in surrogate mode.")

    if max_samples_per_digit is not None and max_samples_per_digit <= 0:
        raise ValueError("max_samples_per_digit must be a positive integer or None.")
    if use_all_digit_examples or max_samples_per_digit is not None:
        source_images, sample_digit_labels, sample_indices = load_all_digit_examples(
            max_samples_per_digit
        )
    else:
        source_images, sample_indices = load_digit_examples()
        sample_digit_labels = np.asarray(list(DIGIT_TO_LETTER), dtype=int)
    test_images, test_digit_labels, test_sample_indices = load_test_digit_examples()
    if mode == "surrogate":
        if not lookup_csv.is_file():
            raise FileNotFoundError(
                f"Surrogate lookup CSV not found: {lookup_csv}. "
                "Pass its location with --lookup-csv."
            )
        df_set, df_reset = load_lookup_table(lookup_csv)
        result_path = RESULTS_PATH / result_variant
        if max_samples_per_digit is not None:
            result_path = result_path / f"first{max_samples_per_digit}_per_digit"
    else:
        df_set = df_reset = None
        result_path = RESULTS_PATH

    weights_dir = result_path / "weights"
    reconstructions_dir = result_path / "reconstructions"
    weights_dir.mkdir(parents=True, exist_ok=True)
    reconstructions_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    training_curves = []

    for input_size in input_sizes:
        input_batch = np.stack(
            [resize_grayscale(image, input_size).ravel() for image in source_images]
        )
        test_input_batch = np.stack(
            [resize_grayscale(image, input_size).ravel() for image in test_images]
        )
        for output_size in output_sizes:
            targets_by_digit = {
                digit: load_letter_target(
                    letter, output_size, target_binarization_threshold
                ).ravel()
                for digit, letter in DIGIT_TO_LETTER.items()
            }
            target_batch = np.stack(
                [targets_by_digit[digit] for digit in sample_digit_labels]
            )
            test_target_batch = np.stack(
                [targets_by_digit[digit] for digit in test_digit_labels]
            )
            if mode == "surrogate":
                (
                    weights,
                    mse_history,
                    lookup_attempt_count,
                    effective_update_count,
                    samples_over_mse_threshold_history,
                    samples_passing_gate_history,
                    lookup_attempts_per_epoch,
                    effective_updates_per_epoch,
                ) = train_surrogate_regression(
                    input_batch,
                    target_batch,
                    steps,
                    learning_rate,
                    df_set,
                    df_reset,
                    seed,
                    sample_mse_threshold,
                    pulse_voltage_range,
                    update_gate,
                    (
                        0.5
                        if target_binarization_threshold is None
                        else target_binarization_threshold
                    ),
                )
                for epoch, epoch_mse in enumerate(mse_history, start=1):
                    training_curves.append(
                        {
                            "input_size": input_size,
                            "output_size": output_size,
                            "epoch": epoch,
                            "pixel_mse": epoch_mse,
                            "samples_over_mse_threshold": (
                                samples_over_mse_threshold_history[epoch - 1]
                            ),
                            "samples_passing_update_gate": (
                                samples_passing_gate_history[epoch - 1]
                            ),
                            (
                                "samples_with_misclassified_pixel"
                                if update_gate == "misclassification"
                                else "samples_passing_mse_gate"
                            ): samples_passing_gate_history[epoch - 1],
                            "lookup_update_attempts": lookup_attempts_per_epoch[
                                epoch - 1
                            ],
                            "effective_weight_updates": effective_updates_per_epoch[
                                epoch - 1
                            ],
                        }
                    )
                max_weight_error = None
            else:
                weights, learning_rate = train_linear_regression(
                    input_batch, target_batch, steps
                )
                reference_weights = target_batch.T @ np.linalg.pinv(input_batch.T)
                max_weight_error = float(np.max(np.abs(weights - reference_weights)))
                lookup_attempt_count = None
                effective_update_count = None

            predictions = input_batch @ weights.T
            pixel_mse = float(np.mean((predictions - target_batch) ** 2))
            test_predictions = test_input_batch @ weights.T
            test_pixel_mse = float(
                np.mean((test_predictions - test_target_batch) ** 2)
            )

            variant = f"input{input_size}_output{output_size}"
            weight_path = weights_dir / f"{variant}.npz"
            np.savez_compressed(
                weight_path,
                W=weights,
                input_size=input_size,
                output_size=output_size,
                digits=np.array(list(DIGIT_TO_LETTER)),
                sample_indices=np.array(sample_indices),
                sample_digit_labels=sample_digit_labels,
                test_sample_indices=np.array(test_sample_indices),
                test_sample_digit_labels=test_digit_labels,
                mode=mode,
                update_gate=update_gate,
                target_binarization_threshold=(
                    np.nan
                    if target_binarization_threshold is None
                    else target_binarization_threshold
                ),
                pulse_voltage_range=np.asarray(
                    [np.nan, np.nan]
                    if pulse_voltage_range is None
                    else pulse_voltage_range,
                    dtype=np.float64,
                ),
            )
            test_representative_rows = {
                digit: np.flatnonzero(test_digit_labels == digit)[:3]
                for digit in DIGIT_TO_LETTER
            }
            for digit, rows in test_representative_rows.items():
                letter = DIGIT_TO_LETTER[digit]
                for sample_number, row in enumerate(rows, start=1):
                    image_path = reconstructions_dir / (
                        f"{variant}_test_digit{digit}_sample{sample_number:02d}_to_{letter}.png"
                    )
                    save_grayscale_image(
                        image_path,
                        test_predictions[row],
                        output_size,
                        target_binarization_threshold,
                    )

            metrics.append(
                {
                    "input_size": input_size,
                    "output_size": output_size,
                    "input_features": input_batch.shape[1],
                    "output_features": target_batch.shape[1],
                    "training_examples": len(source_images),
                    "training_examples_digit_1": int(np.sum(sample_digit_labels == 1)),
                    "training_examples_digit_2": int(np.sum(sample_digit_labels == 2)),
                    "training_examples_digit_3": int(np.sum(sample_digit_labels == 3)),
                    "test_examples": len(test_images),
                    "test_examples_digit_1": int(np.sum(test_digit_labels == 1)),
                    "test_examples_digit_2": int(np.sum(test_digit_labels == 2)),
                    "test_examples_digit_3": int(np.sum(test_digit_labels == 3)),
                    "max_samples_per_digit": max_samples_per_digit,
                    "gradient_steps": steps,
                    "learning_rate": learning_rate,
                    "update_gate": update_gate if mode == "surrogate" else None,
                    "sample_mse_threshold": (
                        sample_mse_threshold
                        if mode == "surrogate"
                        else None
                    ),
                    "target_binarization_threshold": target_binarization_threshold,
                    "pulse_voltage_min_v": (
                        None if pulse_voltage_range is None else pulse_voltage_range[0]
                    ),
                    "pulse_voltage_max_v": (
                        None if pulse_voltage_range is None else pulse_voltage_range[1]
                    ),
                    "surrogate_weight_updates": lookup_attempt_count,
                    "surrogate_lookup_update_attempts": lookup_attempt_count,
                    "surrogate_effective_weight_updates": effective_update_count,
                    "surrogate_samples_over_mse_threshold": (
                        sum(samples_over_mse_threshold_history)
                        if mode == "surrogate"
                        else None
                    ),
                    "surrogate_samples_with_misclassified_pixel": (
                        sum(samples_passing_gate_history)
                        if mode == "surrogate" and update_gate == "misclassification"
                        else None
                    ),
                    "pixel_mse": pixel_mse,
                    "test_pixel_mse": test_pixel_mse,
                    "test_data_scope": "official_imagesTest_digits_1_2_3",
                    "max_abs_weight_error_vs_pseudoinverse": max_weight_error,
                    "sample_indices": (
                        ";".join(map(str, sample_indices))
                        if not use_all_digit_examples
                        else "all_examples_for_digits_1_2_3"
                    ),
                    "training_data_scope": (
                        "all_examples_for_digits_1_2_3"
                        if max_samples_per_digit is None
                        else f"first_{max_samples_per_digit}_per_digit"
                    ),
                    "weights_file": str(weight_path.relative_to(ROOT)),
                }
            )
            status = (
                f"lookup attempts={lookup_attempt_count}, "
                f"effective scalar updates={effective_update_count}, "
                f"samples over MSE threshold="
                f"{sum(samples_over_mse_threshold_history)}, "
                f"samples passing {update_gate} gate="
                f"{sum(samples_passing_gate_history)}, "
                f"test pixel MSE={test_pixel_mse:.8g}"
                if mode == "surrogate"
                else f"max |W - W_pinv|={max_weight_error:.8g}, "
                f"test pixel MSE={test_pixel_mse:.8g}"
            )
            print(
                f"input={input_size}x{input_size}, output={output_size}x{output_size}: "
                f"pixel MSE={pixel_mse:.8g}, {status}"
            )

    metrics_path = result_path / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)
    print(f"Saved metrics to {metrics_path.relative_to(ROOT)}")
    if training_curves:
        curve_path = result_path / "training_curve.csv"
        with curve_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=training_curves[0].keys())
            writer.writeheader()
            writer.writerows(training_curves)
        print(f"Saved training curve to {curve_path.relative_to(ROOT)}")


def parse_args(include_sample_mse_threshold: bool = True) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a single-layer linear regressor from MNIST digits 1/2/3 to C/A/U."
    )
    parser.add_argument("--mode", choices=("software", "surrogate"), default="software")
    parser.add_argument("--input-sizes", nargs="+", type=int, default=[64, 8, 4])
    parser.add_argument("--output-sizes", nargs="+", type=int, default=[64, 8])
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="gradient steps in software mode or epochs in surrogate mode",
    )
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument(
        "--max-samples-per-digit",
        type=int,
        default=None,
        help="limit training to the first N examples of each digit (default: use all)",
    )
    if include_sample_mse_threshold:
        parser.add_argument(
            "--sample-mse-threshold",
            type=float,
            default=0.2,
            help="per-sample MSE gate; binary training also requires a misclassified pixel",
        )
    parser.add_argument("--lookup-csv", type=Path, default=DEFAULT_LOOKUP_PATH)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if (
        args.steps <= 0
        or args.learning_rate <= 0
        or any(size <= 0 for size in args.input_sizes + args.output_sizes)
    ):
        parser.error("steps, learning rate, and image sizes must be positive")
    if include_sample_mse_threshold and (
        not np.isfinite(args.sample_mse_threshold) or args.sample_mse_threshold < 0
    ):
        parser.error("sample MSE threshold must be finite and nonnegative")
    if args.max_samples_per_digit is not None and args.max_samples_per_digit <= 0:
        parser.error("max samples per digit must be a positive integer")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    run_experiments(
        arguments.input_sizes,
        arguments.output_sizes,
        arguments.steps,
        arguments.mode,
        arguments.lookup_csv,
        arguments.learning_rate,
        arguments.seed,
        arguments.sample_mse_threshold,
        max_samples_per_digit=arguments.max_samples_per_digit,
        update_gate=(
            "misclassification" if arguments.mode == "surrogate" else "mse"
        ),
    )