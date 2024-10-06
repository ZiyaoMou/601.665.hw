#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path
import torch
from probs import BackoffAddLambdaLanguageModel, LanguageModel

log = logging.getLogger(Path(__file__).stem)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sentences from a trained n-gram model.")
    parser.add_argument("model_file", type=Path, help="path to the trained language model file")
    parser.add_argument("num_sentences", type=int, help="number of sentences to sample")
    parser.add_argument("--max_length", type=int, default=20, help="maximum sentence length")
    parser.add_argument("--device", type=str, default="cpu", choices=['cpu', 'cuda', 'mps'], help="device to use for PyTorch")
    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # Load the pretrained language model
    model = LanguageModel.load(args.model_file, device=args.device)

    # Sample k sentences
    for i in range(args.num_sentences):
        sentence = model.sample_sentence(max_length=args.max_length)
        print(f"{i+1}: {sentence}")

if __name__ == "__main__":
    main()