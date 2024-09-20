#!/usr/bin/env python3
"""
Determine most similar words in terms of their word embeddings.
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from integerize import Integerizer   # look at integerize.py for more info

# Needed for Python's optional type annotations.
# We've included type annotations and recommend that you do the same, 
# so that mypy (or a similar package) can catch type errors in your code.
from typing import List, Optional

try:
    # PyTorch is your friend. Not *using* it will make your program so slow.
    # And it's also required for this assignment. ;-)
    # So if you comment this block out instead of dealing with it, you're
    # making your own life worse.
    import torch
    import torch.nn as nn
except ImportError:
    print("\nERROR! You need to install Miniconda, then create and activate the nlp-class environment.  See the INSTRUCTIONS file.\n")
    raise


log = logging.getLogger(Path(__file__).stem)  # The only okay global variable.

# Logging is in general a good practice to monitor the behavior of your code
# while it's running. Compared to calling `print`, it provides two benefits.
# 
# - It prints to standard error (stderr), not standard output (stdout) by
#   default.  So these messages will normally go to your screen, even if
#   you have redirected stdout to a file.  And they will not be seen by
#   the autograder, so the autograder won't be confused by them.
# 
# - You can configure how much logging information is provided, by
#   controlling the logging 'level'. You have a few options, like
#   'debug', 'info', 'warning', and 'error'. By setting a global flag,
#   you can ensure that the information you want - and only that info -
#   is printed. As an example:
#        >>> try:
#        ...     rare_word = "prestidigitation"
#        ...     vocab.get_counts(rare_word)
#        ... except KeyError:
#        ...     log.error(f"Word that broke the program: {rare_word}")
#        ...     log.error(f"Current contents of vocab: {vocab.data}")
#        ...     raise  # Crash the program; can't recover.
#        >>> log.info(f"Size of vocabulary is {len(vocab)}")
#        >>> if len(vocab) == 0:
#        ...     log.warning(f"Empty vocab. This may cause problems.")
#        >>> log.debug(f"The values are {vocab}")
#   If we set the log level to be 'INFO', only the log.info, log.warning,
#   and log.error statements will be printed. You can calibrate exactly how 
#   much info you need, and when. None of these pollute stdout with things 
#   that aren't the real 'output' of your program.
# 
# In `parse_args`, we provided two command-line options to control the logging level.
# The default level is 'INFO'. You can lower it to 'DEBUG' if you pass '--verbose'
# and you can raise it to 'WARNING' if you pass '--quiet'.
#
# More info: https://docs.python.org/3/howto/logging.html#logging-basic-tutorial
# 
# In all the starter code for the NLP course, we've elected to create a separate
# logger for each source code file, stored in a variable named log that
# is globally visible throughout the file.  That way, calls like log.info(...)
# will use the logger for the current source code file and thus their output will 
# helpfully show the filename.  You could configure the current file's logger using
# log.basicConfig(...), whereas logging.basicConfig(...) affects all of the loggers.
# The command-line options affect all of the loggers.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("embeddings", type=Path, help="path to word embeddings file")
    parser.add_argument("word", type=str, help="word to look up")
    parser.add_argument("--minus", type=str, default=None)
    parser.add_argument("--plus", type=str, default=None)

    # for verbosity of logging
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet",   dest="logging_level", action="store_const", const=logging.WARNING
    )

    args = parser.parse_args()
    if not args.embeddings.is_file():
        parser.error(f"Embeddings file {args.embeddings} not found")
    if (args.minus is None) != (args.plus is None):  # != is the XOR operation!
        parser.error("Must include both `--plus` and `--minus` or neither")

    return args

class Lexicon:
    """
    Class that manages a lexicon and can compute similarity.

    >>> my_lexicon = Lexicon.from_file(my_file)
    >>> my_lexicon.find_similar_words(bagpipe)
    """

    def __init__(self, integerizer, matrix ) -> None:
        """Load information into coupled word-index mapping and embedding matrix."""
        # FINISH THIS FUNCTION

        # Store your stuff! Both the word-index mapping and the embedding matrix.
        #
        # Do something with this size info?
        # PyTorch's torch.Tensor objects rely on fixed-size arrays in memory.
        # One of the worst things you can do for efficiency is
        # append row-by-row, like you would with a Python list.
        #
        # Probably make the entire list all at once, then convert to a torch.Tensor.
        # Otherwise, make the torch.Tensor and overwrite its contents row-by-row.
        self.words = integerizer
        self.embeddings = matrix

    @classmethod
    def from_file(cls, file: Path) -> Lexicon:
        # FINISH THIS FUNCTION
        integerizer = Integerizer()
        embeddings_list = []
        with open(file) as f:
            first_line = next(f)  # Peel off the special first line.
            for line in f:  # All of the other lines are regular.
                integerizer.add(line.split()[0])  # Add the word to the integerizer.
                embeddings_list.append([ float(x) for x in line.split()[1:]])  # Add the embedding to the list.
        matrix = torch.tensor(embeddings_list)  # Convert the list to a tensor.
        lexicon = Lexicon(integerizer, matrix)  # Maybe put args here. Maybe follow Builder pattern.
        return lexicon

    def find_similar_words(
        self, wb : str, *, plus: Optional[str] = None, minus: Optional[str] = None
    ) -> List[str]:
        """Find most similar words, in terms of embeddings, to a query."""
        # FINISH THIS FUNCTION

        # The star above forces you to use `plus` and `minus` only
        # as named arguments. This helps avoid mixups or readability
        # problems where you forget which comes first.
        #
        # We've also given `plus` and `minus` the type annotation
        # Optional[str]. This means that the argument may be None, or
        # it may be a string. If you don't provide these, it'll automatically
        # use the default value we provided: None.
        if (minus is None) != (plus is None):  # != is the XOR operation!
            raise TypeError("Must include both of `plus` and `minus` or neither.")

        index = self.words.index(wb)
        word_vector = self.embeddings[index]

        if minus: 
            minus_vector = self.embeddings[self.words.index(minus)]
            word_vector = word_vector - minus_vector
        if plus:
            plus_vector = self.embeddings[self.words.index(plus)]
            word_vector = word_vector + plus_vector

        # similarities = torch.matmul(self.embeddings, word_vector) / (
        #     torch.norm(self.embeddings, dim=1) * torch.norm(word_vector)
        # )

        normalized_word_vector = word_vector / torch.norm(word_vector) 
        normalized_embeddings = self.embeddings / torch.norm(self.embeddings, dim=1, keepdim=True)
        similarities = torch.matmul(normalized_embeddings, normalized_word_vector) 

        top_similarities, top_indices = torch.topk(similarities, k=10, largest=True)

        similar_words = [self.words._objects[idx] for idx in top_indices if self.words.index_to_word(idx) != wb]

        return similar_words


def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)
    lexicon = Lexicon.from_file(args.embeddings)
    similar_words = lexicon.find_similar_words(
        args.word, plus=args.plus, minus=args.minus
    )
    print(" ".join(similar_words))  # print all words on one line, separated by spaces


if __name__ == "__main__":
    main()
