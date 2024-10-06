#!/usr/bin/env python3
"""
Computes the total log probability of the sequences of tokens in each file,
according to a given smoothed trigram model.  
"""
import argparse
import logging
import math
from pathlib import Path
import torch
# import matplotlib.pyplot as plt
# import numpy as np
# import statsmodels.api as sm

from probs import Wordtype, LanguageModel, num_tokens, read_trigrams, BackoffAddLambdaLanguageModel

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model1",
        type=Path,
        help="path to the trained model for category 1",
    )
    parser.add_argument(
        "model2",
        type=Path,
        help="path to the trained model for category 2",
    )
    parser.add_argument(
        "prior1",
        type=float,
        help="prior probability of category 1 (e.g., 0.7 means P(category1)=0.7 and P(category2)=0.3)",
    )
    parser.add_argument(
        "test_files",
        type=Path,
        nargs="*"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=['cpu','cuda','mps'],
        help="device to use for PyTorch (cpu or cuda, or mps if you are on a mac)"
    )

    # for verbosity of logging
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet",   dest="logging_level", action="store_const", const=logging.WARNING
    )

    return parser.parse_args()


def file_log_prob(file: Path, lm: LanguageModel) -> float:
    """The file contains one sentence per line. Return the total
    log-probability of all these sentences, under the given language model.
    (This is a natural log, as for all our internal computations.)
    """
    log_prob = 0.0

    x: Wordtype; y: Wordtype; z: Wordtype    # type annotation for loop variables below
    for (x, y, z) in read_trigrams(file, lm.vocab):
        log_prob += lm.log_prob(x, y, z)  # log p(z | xy)

        # If the factor p(z | xy) = 0, then it will drive our cumulative file 
        # probability to 0 and our cumulative log_prob to -infinity.  In 
        # this case we can stop early, since the file probability will stay 
        # at 0 regardless of the remaining tokens.
        if log_prob == -math.inf: break 

        # Why did we bother stopping early?  It could occasionally
        # give a tiny speedup, but there is a more subtle reason -- it
        # avoids a ZeroDivisionError exception in the unsmoothed case.
        # If xyz has never been seen, then perhaps yz hasn't either,
        # in which case p(next token | yz) will be 0/0 if unsmoothed.
        # We can avoid having Python attempt 0/0 by stopping early.
        # (Conceptually, 0/0 is an indeterminate quantity that could
        # have any value, and clearly its value doesn't matter here
        # since we'd just be multiplying it by 0.)

    return log_prob

def classify_file(file: Path, lm1: LanguageModel, lm2: LanguageModel, prior1: float) -> str:
    """Classifies the file as category 1 or category 2 using the two models and the prior."""
    prior2 = 1 - prior1

    # Compute log-probabilities for both classes
    log_prob1 = file_log_prob(file, lm1) + math.log(prior1)
    log_prob2 = file_log_prob(file, lm2) + math.log(prior2)
    # Classify based on maximum posterior probability
    if log_prob1 > log_prob2:
        return "model1", log_prob1
    else:
        return "model2", log_prob2

def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    # Specify hardware device where all tensors should be computed and
    # stored.  This will give errors unless you have such a device
    # (e.g., 'gpu' will work in a Kaggle Notebook where you have
    # turned on GPU acceleration).
    if args.device == 'mps':
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                logging.critical("MPS not available because the current PyTorch install was not "
                    "built with MPS enabled.")
            else:
                logging.critical("MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine.")
            exit(1)
    torch.set_default_device(args.device)
        
    log.info("Testing...")
    
    lm1 = LanguageModel.load(args.model1, device=args.device)
    lm2 = LanguageModel.load(args.model2, device=args.device)
    files = args.test_files
    model1_count = 0
    error_count = 0
    total_log_prob_lm1 = 0.0
    total_log_prob_lm2 = 0.0
    for file in files:
        model_type, log_prob = classify_file(file, lm1, lm2, args.prior1)
        if((model_type == "model1" and file.name.split(".")[0] != "gen") or (model_type == "model2" and file.name.split(".")[0] != "spam")):
            error_count += 1
        if(model_type == "model1"):
            model1_count += 1
            total_log_prob_lm1 += log_prob
            print(f"{args.model1}\t{file}")
        else :
            # continue
            total_log_prob_lm2 += log_prob
            print(f"{args.model2}\t{file}")
    print(f"{model1_count} files were more probably from {args.model1}({model1_count/len(files)*100:.2f}%)")
    print(f"{len(files) - model1_count} files were more probably from {args.model2}({(len(files) - model1_count)/len(files)*100:.2f}%)")
    print(f"Error rate: {error_count/len(files)*100:.2f}%")
    # We use natural log for our internal computations and that's
    # the kind of log-probability that file_log_prob returns.
    # We'll print that first.



    # # # But cross-entropy is conventionally measured in bits: so when it's
    # # # time to print cross-entropy, we convert log base e to log base 2, 
    # # # by dividing by log(2).

    # lm1_bits = -total_log_prob_lm1 / math.log(2)   # convert to bits of surprisal
    # lm2_bits = -total_log_prob_lm2 / math.log(2)   # convert to bits of surprisal

    # # # # We also divide by the # of tokens (including EOS tokens) to get
    # # # # bits per token.  (The division happens within the print statement.)
    # lm1_tokens = sum(test_file.name.split(".")[0] == "gen" and num_tokens(test_file) for test_file in args.test_files)
    # lm2_tokens = sum(test_file.name.split(".")[0] == "spam" and num_tokens(test_file) for test_file in args.test_files)
    # print( lm1_tokens + lm2_tokens == (sum(num_tokens(test_file) for test_file in args.test_files)))
    # # print(total_log_prob_lm1, total_log_prob_lm2)
    # # print(f"Gen Overall cross-entropy:\t{lm1_bits / lm1_tokens:.2f} bits per token")
    # # print(f'Perplexity: {2**(lm1_bits / lm1_tokens):.5f}')
    # # print(f"Spam Overall cross-entropy:\t{lm2_bits / lm2_tokens:.2f} bits per token")
    # # print(f'Perplexity: {2**(lm2_bits / lm2_tokens):.5f}')
    # print(f"Overall cross-entropy:\t{(lm1_bits + lm2_bits) / (lm1_tokens + lm2_tokens):.2f} bits per token")

    # file_lengths = np.array(file_lengths)
    # print(file_lengths)
    # cross_entropy_values = np.array(cross_entropy_values)

    # lowess = sm.nonparametric.lowess(cross_entropy_values, file_lengths, frac=0.3)

    # log_file_lengths = np.log([eval(file_length) for file_length in file_lengths])
    # x_vals = np.linspace(min(log_file_lengths), max(log_file_lengths), 100)
    # y_vals = np.interp(x_vals, np.log(lowess[:, 0]), lowess[:, 1])

    # plt.figure(figsize=(10, 6))
    # plt.scatter(log_file_lengths, cross_entropy_values, label='Raw Data', color='blue')
    # plt.plot(x_vals, y_vals, label='Loess Smoothing', color='red')

    # plt.xscale('log')  # 设置 x 轴为对数刻度
    # plt.xlabel('Log(File Length)')
    # plt.ylabel('Cross-Entropy')
    # plt.title('Cross-Entropy vs Log(File Length) with Loess Smoothing when λ = λ∗ ')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
if __name__ == "__main__":
    main()

