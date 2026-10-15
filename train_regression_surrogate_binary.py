"""Run surrogate regression with black-and-white letter targets."""

from train_regression_surrogate import parse_args, run_experiments
# & .\.venv\Scripts\python.exe train_regression_surrogate_binary.py --mode surrogate --lookup-csv .\model_run_all_voltages.csv --input-sizes 8 --output-sizes 8 --steps 10000


BINARY_THRESHOLD = 0.5
RESULT_VARIANT = "surrogate_binary"


def main() -> None:
    arguments = parse_args()
    if arguments.mode != "surrogate":
        raise SystemExit("This runner requires --mode surrogate.")

    run_experiments(
        arguments.input_sizes,
        arguments.output_sizes,
        arguments.steps,
        arguments.mode,
        arguments.lookup_csv,
        arguments.learning_rate,
        arguments.seed,
        arguments.sample_mse_threshold,
        result_variant=RESULT_VARIANT,
        target_binarization_threshold=BINARY_THRESHOLD,
        update_gate="misclassification",
        use_all_digit_examples=True,
        max_samples_per_digit=arguments.max_samples_per_digit,
    )


if __name__ == "__main__":
    main()