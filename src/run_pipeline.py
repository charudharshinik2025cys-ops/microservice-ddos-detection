"""Run synthetic generation, training, and evaluation in order."""
from .generate_data import generate
from .train import main as train
from .evaluate import main as evaluate


def main():
    generate()
    train()
    evaluate()


if __name__ == "__main__":
    main()
