#!/usr/bin/env python3
import argparse
import logging
import math
from pathlib import Path
import torch
from probs import Wordtype, LanguageModel, num_tokens, read_trigrams
from jiwer import wer

log = logging.getLogger(Path(__file__).stem)  # Logging setup

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute WER from transcription files.")
    parser.add_argument("model", type=Path, help="Path to the trained model")
    parser.add_argument("transcription_files", type=Path, nargs="*", help="List of transcription files")
    parser.add_argument("--device", type=str, default="cpu", choices=['cpu', 'cuda', 'mps'], help="Device to use for PyTorch")
    
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG)
    verbosity.add_argument("-q", "--quiet", dest="logging_level", action="store_const", const=logging.WARNING)

    return parser.parse_args()


def word_error_rate(predicted: str, reference: str) -> tuple:
    """Compute word error rate (WER) using JiWER."""
    # Using JiWER for WER calculation
    wer_value = wer(reference, predicted)
    errors = wer_value * len(reference.split())
    return errors, len(reference.split())


def file_log_prob(file: Path, lm: LanguageModel) -> float:
    log_prob = 0.0
    for (x, y, z) in read_trigrams(file, lm.vocab):
        log_prob += lm.log_prob(x, y, z)
        if log_prob == -math.inf:
            break  # Stop early if cumulative log-prob hits -inf
    return log_prob


def parse_transcription_file(file: Path):
    """Parses transcription candidates from the given file."""
    with open(file) as f:
        lines = f.readlines()
    
    correct_transcription = lines[0].strip().split(maxsplit=1)[1]
    candidates = []
    for line in lines[1:]:
        parts = line.strip().split(maxsplit=3)
        prob = float(parts[0])
        log_score = float(parts[1])
        transcription = parts[3]
        candidates.append((prob, log_score, transcription))
    
    return correct_transcription, candidates


def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    # Set device for PyTorch
    if args.device == 'mps' and not torch.backends.mps.is_available():
        logging.critical("MPS not available on this device.")
        exit(1)
    torch.set_default_device(args.device)
        
    log.info("Testing...")
    lm = LanguageModel.load(args.model, device=args.device)
    
    total_errors = 0
    total_words = 0

    for transcription_file in args.transcription_files:
        correct_transcription, candidates = parse_transcription_file(transcription_file)

        # Find the best candidate
        best_candidate = None
        best_log_prob_sum = float("inf")

        for prob, log_score, transcription in candidates:
            log_prob_lm = file_log_prob(transcription_file, lm)
            combined_log_prob = log_prob_lm + log_score

            if combined_log_prob < best_log_prob_sum:
                best_log_prob_sum = combined_log_prob
                best_candidate = transcription

        # Compute WER
        errors, words = word_error_rate(best_candidate, correct_transcription)
        error_rate = errors / words
        print(f"{error_rate:.3f}\t{transcription_file.name}")

        total_errors += errors
        total_words += words

    overall_wer = total_errors / total_words
    print(f"{overall_wer:.3f}\tOVERALL")


if __name__ == "__main__":
    main()