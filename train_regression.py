import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "Inputs" / "MNISTdataset0123.mat"
OUTPUTS_PATH = ROOT / "Outputs"
RESULTS_PATH = OUTPUTS_PATH / "regression_results"
DIGIT_TO_LETTER = {1: "c", 2: "a", 3: "u"}


def resize_grayscale(image: np.ndarray, size: int) -> np.ndarray:
    pixels = np.rint(np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    resized = Image.fromarray(pixels).resize(
        (size, size), Image.Resampling.BILINEAR
    )
    return np.asarray(resized, dtype=np.float64) / 255.0


def load_digit_examples(
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

    sample_indices = []
    for digit in DIGIT_TO_LETTER:
        matching_indices = np.flatnonzero(labels == digit)
        if matching_indices.size == 0:
            raise ValueError(f"No training example found for digit {digit}.")
        if max_samples_per_digit is not None:
            matching_indices = matching_indices[:max_samples_per_digit]
        sample_indices.extend(matching_indices.tolist())

    sample_indices.sort()
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


def load_letter_target(letter: str, size: int) -> np.ndarray:
    target_path = OUTPUTS_PATH / f"{letter}_{size}.png"
    if not target_path.is_file():
        raise FileNotFoundError(f"Missing target image: {target_path}")
    with Image.open(target_path) as image:
        grayscale = image.convert("L").resize(
            (size, size), Image.Resampling.BILINEAR
        )
        return np.asarray(grayscale, dtype=np.float64) / 255.0


def train_linear_regression(
    inputs: np.ndarray, targets: np.ndarray, steps: int
) -> tuple[np.ndarray, float]:
    sample_count = inputs.shape[0]
    if inputs.shape[1] > sample_count:
        gram = inputs @ inputs.T
        largest_eigenvalue = np.linalg.eigvalsh(gram / sample_count).max()
        if largest_eigenvalue <= 0:
            raise ValueError("Input examples have no nonzero pixel values.")
        learning_rate = 1.0 / largest_eigenvalue
        coefficients = np.zeros((targets.shape[1], sample_count), dtype=np.float64)
        for _ in range(steps):
            errors = gram @ coefficients.T - targets
            coefficients -= learning_rate * errors.T / sample_count
        return coefficients @ inputs, learning_rate

    input_covariance = inputs.T @ inputs / sample_count
    target_input_covariance = targets.T @ inputs / sample_count
    largest_eigenvalue = np.linalg.eigvalsh(input_covariance).max()
    if largest_eigenvalue <= 0:
        raise ValueError("Input examples have no nonzero pixel values.")
    learning_rate = 1.0 / largest_eigenvalue

    weights = np.zeros((targets.shape[1], inputs.shape[1]), dtype=np.float64)
    for _ in range(steps):
        weights -= learning_rate * (
            weights @ input_covariance - target_input_covariance
        )
    return weights, learning_rate


def save_grayscale_image(path: Path, pixels: np.ndarray, size: int) -> None:
    image = np.rint(np.clip(pixels.reshape(size, size), 0.0, 1.0) * 255)
    Image.fromarray(image.astype(np.uint8)).save(path)


def run_experiments(
    input_sizes: list[int],
    output_sizes: list[int],
    steps: int,
    max_samples_per_digit: int | None = None,
) -> None:
    source_images, sample_digit_labels, sample_indices = load_digit_examples(
        max_samples_per_digit
    )
    test_images, test_digit_labels, test_sample_indices = load_test_digit_examples()
    result_path = RESULTS_PATH
    if max_samples_per_digit is not None:
        result_path = RESULTS_PATH / f"first{max_samples_per_digit}_per_digit"
    weights_dir = result_path / "weights"
    reconstructions_dir = result_path / "reconstructions"
    weights_dir.mkdir(parents=True, exist_ok=True)
    reconstructions_dir.mkdir(parents=True, exist_ok=True)
    metrics = []

    for input_size in input_sizes:
        input_batch = np.stack(
            [resize_grayscale(image, input_size).ravel() for image in source_images]
        )
        test_input_batch = np.stack(
            [resize_grayscale(image, input_size).ravel() for image in test_images]
        )
        for output_size in output_sizes:
            targets_by_digit = {
                digit: load_letter_target(letter, output_size).ravel()
                for digit, letter in DIGIT_TO_LETTER.items()
            }
            target_batch = np.stack(
                [targets_by_digit[digit] for digit in sample_digit_labels]
            )
            test_target_batch = np.stack(
                [targets_by_digit[digit] for digit in test_digit_labels]
            )
            weights, learning_rate = train_linear_regression(
                input_batch, target_batch, steps
            )
            predictions = input_batch @ weights.T
            pixel_mse = float(np.mean((predictions - target_batch) ** 2))
            test_predictions = test_input_batch @ weights.T
            test_pixel_mse = float(
                np.mean((test_predictions - test_target_batch) ** 2)
            )
            if input_batch.shape[1] <= 64:
                input_covariance = input_batch.T @ input_batch / len(input_batch)
                target_input_covariance = target_batch.T @ input_batch / len(input_batch)
                reference_weights = target_input_covariance @ np.linalg.pinv(
                    input_covariance
                )
                max_weight_error = float(np.max(np.abs(weights - reference_weights)))
            else:
                max_weight_error = None

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
                max_samples_per_digit=(
                    -1 if max_samples_per_digit is None else max_samples_per_digit
                ),
            )
            representative_rows = {
                digit: np.flatnonzero(test_digit_labels == digit)[:3]
                for digit in DIGIT_TO_LETTER
            }
            for digit, rows in representative_rows.items():
                letter = DIGIT_TO_LETTER[digit]
                for sample_number, row in enumerate(rows, start=1):
                    image_path = reconstructions_dir / (
                        f"{variant}_test_digit{digit}_sample{sample_number:02d}_to_{letter}.png"
                    )
                    save_grayscale_image(
                        image_path, test_predictions[row], output_size
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
                    "pixel_mse": pixel_mse,
                    "test_pixel_mse": test_pixel_mse,
                    "test_data_scope": "official_imagesTest_digits_1_2_3",
                    "max_abs_weight_error_vs_pseudoinverse": max_weight_error,
                    "sample_indices": (
                        "all_examples_for_digits_1_2_3"
                        if max_samples_per_digit is None
                        else f"first_{max_samples_per_digit}_per_digit"
                    ),
                    "training_data_scope": (
                        "all_examples_for_digits_1_2_3"
                        if max_samples_per_digit is None
                        else f"first_{max_samples_per_digit}_per_digit"
                    ),
                    "weights_file": str(weight_path.relative_to(ROOT)),
                }
            )
            print(
                f"input={input_size}x{input_size}, output={output_size}x{output_size}: "
                f"pixel MSE={pixel_mse:.8g}, "
                f"test pixel MSE={test_pixel_mse:.8g}, "
                f"{f'max |W - W_pinv|={max_weight_error:.8g}' if max_weight_error is not None else 'pseudoinverse comparison skipped for this input size'}"
            )

    metrics_path = result_path / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)
    print(f"Saved metrics to {metrics_path.relative_to(ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a single-layer linear regressor from MNIST digits 1/2/3 to C/A/U."
    )
    parser.add_argument("--input-sizes", nargs="+", type=int, default=[64, 8, 4])
    parser.add_argument("--output-sizes", nargs="+", type=int, default=[64, 8])
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument(
        "--max-samples-per-digit",
        type=int,
        default=None,
        help="limit training to the first N examples of each digit (default: use all)",
    )
    args = parser.parse_args()
    if (
        args.steps <= 0
        or any(size <= 0 for size in args.input_sizes + args.output_sizes)
        or (args.max_samples_per_digit is not None and args.max_samples_per_digit <= 0)
    ):
        parser.error("steps and image sizes must be positive integers")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    run_experiments(
        arguments.input_sizes,
        arguments.output_sizes,
        arguments.steps,
        arguments.max_samples_per_digit,
    )