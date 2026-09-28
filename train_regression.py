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


def save_grayscale_image(path: Path, pixels: np.ndarray, size: int) -> None:
    image = np.rint(np.clip(pixels.reshape(size, size), 0.0, 1.0) * 255)
    Image.fromarray(image.astype(np.uint8)).save(path)


def run_experiments(input_sizes: list[int], output_sizes: list[int], steps: int) -> None:
    source_images, sample_indices = load_digit_examples()
    weights_dir = RESULTS_PATH / "weights"
    reconstructions_dir = RESULTS_PATH / "reconstructions"
    weights_dir.mkdir(parents=True, exist_ok=True)
    reconstructions_dir.mkdir(parents=True, exist_ok=True)
    metrics = []

    for input_size in input_sizes:
        input_batch = np.stack(
            [resize_grayscale(image, input_size).ravel() for image in source_images]
        )
        for output_size in output_sizes:
            target_batch = np.stack(
                [
                    load_letter_target(DIGIT_TO_LETTER[digit], output_size).ravel()
                    for digit in DIGIT_TO_LETTER
                ]
            )
            weights, learning_rate = train_linear_regression(
                input_batch, target_batch, steps
            )
            predictions = input_batch @ weights.T
            pixel_mse = float(np.mean((predictions - target_batch) ** 2))
            reference_weights = target_batch.T @ np.linalg.pinv(input_batch.T)
            max_weight_error = float(np.max(np.abs(weights - reference_weights)))

            variant = f"input{input_size}_output{output_size}"
            weight_path = weights_dir / f"{variant}.npz"
            np.savez_compressed(
                weight_path,
                W=weights,
                input_size=input_size,
                output_size=output_size,
                digits=np.array(list(DIGIT_TO_LETTER)),
                sample_indices=np.array(sample_indices),
            )
            for row, digit in enumerate(DIGIT_TO_LETTER):
                letter = DIGIT_TO_LETTER[digit]
                image_path = reconstructions_dir / f"{variant}_digit{digit}_to_{letter}.png"
                save_grayscale_image(image_path, predictions[row], output_size)

            metrics.append(
                {
                    "input_size": input_size,
                    "output_size": output_size,
                    "input_features": input_batch.shape[1],
                    "output_features": target_batch.shape[1],
                    "training_examples": len(DIGIT_TO_LETTER),
                    "gradient_steps": steps,
                    "learning_rate": learning_rate,
                    "pixel_mse": pixel_mse,
                    "max_abs_weight_error_vs_pseudoinverse": max_weight_error,
                    "sample_indices": ";".join(map(str, sample_indices)),
                    "weights_file": str(weight_path.relative_to(ROOT)),
                }
            )
            print(
                f"input={input_size}x{input_size}, output={output_size}x{output_size}: "
                f"pixel MSE={pixel_mse:.8g}, "
                f"max |W - W_pinv|={max_weight_error:.8g}"
            )

    metrics_path = RESULTS_PATH / "metrics.csv"
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
    parser.add_argument("--output-sizes", nargs="+", type=int, default=[64, 16])
    parser.add_argument("--steps", type=int, default=20_000)
    args = parser.parse_args()
    if args.steps <= 0 or any(size <= 0 for size in args.input_sizes + args.output_sizes):
        parser.error("steps and image sizes must be positive integers")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    run_experiments(arguments.input_sizes, arguments.output_sizes, arguments.steps)