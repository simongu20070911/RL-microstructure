"""Compatibility wrapper for the relocated training entry point."""

from scripts.training.run_two_sided_training import main  # type: ignore

if __name__ == "__main__":
    main()
