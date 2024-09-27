#!/usr/bin/env python3
"""
Builds a vocabulary of all types that appear "often enough" in a 
training corpus, including the special types OOV and EOS.
The vocabulary is saved as a text file where each line is a word.  
Tokenization is handled by probs.py (currently tokenization at whitespace).
"""
import argparse
import sys
from typing import Set, Counter
from collections import Counter
from pathlib import Path

from probs import Wordtype, EOS, OOV, read_tokens


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "documents",
        nargs="+",
        type=Path,
        help="A list of text documents from which to extract the vocabulary")
    parser.add_argument(
        "--output",
        type=Path,
        default="vocab.txt",
        help="The file to save the vocabulary to"
    )
    parser.add_argument(
        "--threshold",
        default=1,
        type=int,
        help="The minimum number of times a word has to appear for it to be included in the vocabulary (default 1)")

    return parser.parse_args()

def build_vocab(*files: Path, threshold: int) -> Set[str]:
    word_counts: Counter[Wordtype] = Counter()  # count of each word
    for file in files:
        token: Wordtype   # type annotation for loop variable below
        for token in read_tokens(file):
            word_counts[token] += 1

    vocab = set(w for w in word_counts if word_counts[w] >= threshold)
    vocab |= {  # the |= operator modifies vocab by taking its union with this set of size 2
        OOV,
        EOS,
    }  # We make sure that EOS is in the vocab, even if read_tokens returns it too few times.
       # But BOS is not in the vocab: it is never a possible outcome, only a context.

    sys.stderr.write(f"Vocabulary size is {len(vocab)} types including OOV and EOS\n")
    return vocab


def save_vocab(vocab: Set[str], output: Path):
    with open(output, "wt") as f:
        for word in vocab:
            print(word, file=f)


def main():
    """
    A vocab file is just a list of words.

    Before using, change this script to be executable on unix systems via chmod +x build_vocab.py.
    Alternatively, use python3 build_vocab.py instead of ./build_vocab.py in the following example.

    Example usage:

        Build a vocab file out of the union of words in spam and gen

        ./build_vocab.py ../data/gen_spam/train/gen ../data/gen_spam/train/spam --threshold 3 --output vocab-genspam.txt 

        After which you should see the following saved vocab file:

        vocab-genspam.txt
    """
    args = parse_args()
    vocab = build_vocab(*args.documents, threshold=args.threshold)
    save_vocab(vocab, args.output)

if __name__ == '__main__':
    main()
