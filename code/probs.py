#!/usr/bin/env python3
# CS465 at Johns Hopkins University.
# Module to estimate n-gram probabilities.

# Updated by Jason Baldridge <jbaldrid@mail.utexas.edu> for use in NLP
# course at UT Austin. (9/9/2008)

# Modified by Mozhi Zhang <mzhang29@jhu.edu> to add the new log linear model
# with word embeddings.  (2/17/2016)

# Refactored by Arya McCarthy <xkcd@jhu.edu> because inheritance is cool
# and so is separating business logic from other stuff.  (9/19/2019)

# Patched by Arya McCarthy <arya@jhu.edu> to fix a counting issue that
# evidently was known pre-2016 but then stopped being handled?

# Further refactoring by Jason Eisner <jason@cs.jhu.edu> 
# and Brian Lu <zlu39@jhu.edu>.  (9/26/2021)

from __future__ import annotations

import logging
import math
import sys

from pathlib import Path

import torch
import pickle
from torch import nn
from torch import optim
from jaxtyping import Float
from typeguard import typechecked
from typing import Counter, Dict
from collections import Counter
from tqdm import tqdm
import numpy as np
from collections import deque

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

##### TYPE DEFINITIONS (USED FOR TYPE ANNOTATIONS)
from typing import Iterable, List, Optional, Set, Tuple, Union

Wordtype = str  # if you decide to integerize the word types, then change this to int
Vocab    = Set[Wordtype]
Zerogram = Tuple[()]
Unigram  = Tuple[Wordtype]
Bigram   = Tuple[Wordtype, Wordtype]
Trigram  = Tuple[Wordtype, Wordtype, Wordtype]
Ngram    = Union[Zerogram, Unigram, Bigram, Trigram]
Vector   = List[float]
TorchScalar = Float[torch.Tensor, ""] # a torch.Tensor with no dimensions, i.e., a scalar


##### CONSTANTS
BOS: Wordtype = "BOS"  # special word type for context at Beginning Of Sequence
EOS: Wordtype = "EOS"  # special word type for observed token at End Of Sequence
OOV: Wordtype = "OOV"  # special word type for all Out-Of-Vocabulary words
OOL: Wordtype = "OOL"  # special word type whose embedding is used for OOV and all other Out-Of-Lexicon words


##### UTILITY FUNCTIONS FOR CORPUS TOKENIZATION

def read_tokens(file: Path, vocab: Optional[Vocab] = None) -> Iterable[Wordtype]:
    """Iterator over the tokens in file.  Tokens are whitespace-delimited.
    If vocab is given, then tokens that are not in vocab are replaced with OOV."""

    # OPTIONAL SPEEDUP: You may want to modify this to integerize the
    # tokens, using integerizer.py as in previous homeworks.
    # In that case, redefine `Wordtype` from `str` to `int`.

    # PYTHON NOTE: This function uses `yield` to return the tokens one at
    # a time, rather than constructing the whole sequence and using
    # `return` to return it.
    #
    # A function that uses `yield` is called a "generator."  As with other
    # iterators, it computes new values only as needed.  The sequence is
    # never fully constructed as an single object in memory.
    #
    # You can iterate over the yielded sequence, for example, like this:
    #      for token in read_tokens(my_file, vocab):
    #          process(token)
    # Whenever the `for` loop needs another token, read_tokens magically picks up 
    # where it left off and continues running until the next `yield` statement.

    with open(file) as f:
        for line in f:
            for token in line.split():
                if vocab is None or token in vocab:
                    yield token
                else:
                    yield OOV  # replace this out-of-vocabulary word with OOV
            yield EOS  # Every line in the file implicitly ends with EOS.


def num_tokens(file: Path) -> int:
    """Give the number of tokens in file, including EOS."""
    return sum(1 for _ in read_tokens(file))


def read_trigrams(file: Path, vocab: Vocab) -> Iterable[Trigram]:
    """Iterator over the trigrams in file.  Each triple (x,y,z) is a token z
    (possibly EOS) with a left context (x,y)."""
    x, y = BOS, BOS
    for z in read_tokens(file, vocab):
        yield (x, y, z)
        if z == EOS:
            x, y = BOS, BOS  # reset for the next sequence in the file (if any)
        else:
            x, y = y, z  # shift over by one position.


def draw_trigrams_forever(file: Path, 
                          vocab: Vocab, 
                          randomize: bool = False) -> Iterable[Trigram]:
    """Infinite iterator over trigrams drawn from file.  We iterate over
    all the trigrams, then do it again ad infinitum.  This is useful for 
    SGD training.  
    
    If randomize is True, then randomize the order of the trigrams each time.  
    This is more in the spirit of SGD, but the randomness makes the code harder to debug, 
    and forces us to keep all the trigrams in memory at once.
    """
    trigrams = read_trigrams(file, vocab)
    if not randomize:
        import itertools
        return itertools.cycle(trigrams)  # repeat forever
    else:
        import random
        pool = tuple(trigrams)   
        while True:
            for trigram in random.sample(pool, len(pool)):
                yield trigram

##### READ IN A VOCABULARY (e.g., from a file created by build_vocab.py)

def read_vocab(vocab_file: Path) -> Vocab:
    vocab: Vocab = set()
    with open(vocab_file, "rt") as f:
        for line in f:
            word = line.strip()
            vocab.add(word)
    log.info(f"Read vocab of size {len(vocab)} from {vocab_file}")
    return vocab

##### LANGUAGE MODEL PARENT CLASS

class LanguageModel:

    def __init__(self, vocab: Vocab):
        super().__init__()

        self.vocab = vocab
        self.progress = 0   # To print progress.

        self.event_count:   Counter[Ngram] = Counter()  # numerator c(...) function.
        self.context_count: Counter[Ngram] = Counter()  # denominator c(...) function.
        # In this program, the argument to the counter should be an Ngram, 
        # which is always a tuple of Wordtypes, never a single Wordtype:
        # Zerogram: context_count[()]
        # Bigram:   context_count[(x,y)]   or equivalently context_count[x,y]
        # Unigram:  context_count[(y,)]    or equivalently context_count[y,]
        # but not:  context_count[(y)]     or equivalently context_count[y]  
        #             which incorrectly looks up a Wordtype instead of a 1-tuple

    @property
    def vocab_size(self) -> int:
        assert self.vocab is not None
        return len(self.vocab)

    # We need to collect two kinds of n-gram counts.
    # To compute p(z | xy) for a trigram xyz, we need c(xy) for the 
    # denominator and c(yz) for the backed-off numerator.  Both of these 
    # look like bigram counts ... but they are not quite the same thing!
    #
    # For a sentence of length N, we are iterating over trigrams xyz where
    # the position of z falls in 1 ... N+1 (so z can be EOS but not BOS),
    # and therefore
    # the position of y falls in 0 ... N   (so y can be BOS but not EOS).
    # 
    # When we write c(yz), we are counting *events z* with *context* y:
    #         c(yz) = |{i in [1, N]: w[i-1] w[i] = yz}|
    # We keep these "event counts" in `event_count` and use them in the numerator.
    # Notice that z=BOS is not possible (BOS is not a possible event).
    # 
    # When we write c(xy), we are counting *all events* with *context* xy:
    #         c(xy) = |{i in [1, N]: w[i-2] w[i-1] = xy}|
    # We keep these "context counts" in `context_count` and use them in the denominator.
    # Notice that y=EOS is not possible (EOS cannot appear in the context).
    #
    # In short, c(xy) and c(yz) count the training bigrams slightly differently.  
    # Likewise, c(y) and c(z) count the training unigrams slightly differently.
    #
    # Note: For bigrams and unigrams that don't include BOS or EOS -- which
    # is most of them! -- `event_count` and `context_count` will give the
    # same value.  So you could save about half the memory if you were
    # careful to store those cases only once.  (How?)  That would make the
    # code slightly more complicated, but would be worth it in a real system.

    def count_trigram_events(self, trigram: Trigram) -> None:
        """Record one token of the trigram and also of its suffixes (for backoff)."""
        (x, y, z) = trigram
        self.event_count[(x, y, z )] += 1
        self.event_count[   (y, z )] += 1
        self.event_count[      (z,)] += 1  # the comma is necessary to make this a tuple
        self.event_count[        ()] += 1

    def count_trigram_contexts(self, trigram: Trigram) -> None:
        """Record one token of the trigram's CONTEXT portion, 
        and also the suffixes of that context (for backoff)."""
        (x, y, _) = trigram    # we don't care about z
        self.context_count[(x, y )] += 1
        self.context_count[   (y,)] += 1
        self.context_count[     ()] += 1

    def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """Computes an estimate of the trigram log probability log p(z | x,y)
        according to the language model.  The log_prob is what we need to compute
        cross-entropy and to train the model.  It is also unlikely to underflow,
        in contrast to prob.  In many models, we can compute the log_prob directly, 
        rather than first computing the prob and then calling math.log."""
        class_name = type(self).__name__
        if class_name == LanguageModel.__name__:
            raise NotImplementedError("You shouldn't be calling log_prob on an instance of LanguageModel, but on an instance of one of its subclasses.")
        raise NotImplementedError(
            f"{class_name}.log_prob is not implemented yet (you should override LanguageModel.log_prob)"
        )

    def save(self, model_path: Path) -> None:
        log.info(f"Saving model to {model_path}")
        torch.save(self, model_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            # torch.save is similar to pickle.dump but handles tensors too
        log.info(f"Saved model to {model_path}")

    @classmethod
    def load(cls, model_path: Path, device: str = 'cpu') -> "LanguageModel":
        log.info(f"Loading model from {model_path}")
        model = torch.load(model_path, map_location=device)
            # torch.load is similar to pickle.load but handles tensors too
            # map_location allows loading tensors on different device than saved
        if not isinstance(model, cls):
            raise ValueError(f"Type Error: expected object of type {cls} but got {type(result)} from file {model_path}")
        log.info(f"Loaded model from {model_path}")
        return model

    def train(self, file: Path) -> None:
        """Create vocabulary and store n-gram counts.  In subclasses, we might
        override this with a method that computes parameters instead of counts."""

        log.info(f"Training from corpus {file}")

        # Clear out any previous training.
        self.event_count   = Counter()
        self.context_count = Counter()

        for trigram in read_trigrams(file, self.vocab):
            self.count_trigram_events(trigram)
            self.count_trigram_contexts(trigram)
            self.show_progress()

        sys.stderr.write("\n")  # done printing progress dots "...."
        log.info(f"Finished counting {self.event_count[()]} tokens")

    def show_progress(self, freq: int = 5000) -> None:
        """Print a dot to stderr every 5000 calls (frequency can be changed)."""
        self.progress += 1
        if self.progress % freq == 1:
            sys.stderr.write(".")

    def get_next_word_probs(self, x: Wordtype, y: Wordtype) -> torch.Tensor:
        """Get the probability distribution for the next word given (x, y) context."""
        vocab_list = list(self.vocab)  # Convert vocab to a list for indexing
        probs = torch.zeros(len(vocab_list))

        # Compute the probability of each word z in the vocabulary given context (x, y)
        for i, z in enumerate(vocab_list):
            probs[i] = self.prob(x, y, z)

        # Normalize the probabilities in case they don't sum to 1 (should be rare due to smoothing)
        probs /= probs.sum()

        return probs

    def sample_from_distribution(self, probs: torch.Tensor) -> Wordtype:
        """Sample the next word from the provided probability distribution."""
        vocab_list = list(self.vocab)
        sampled_index = torch.multinomial(probs, 1).item()  # Sample from the distribution
        return vocab_list[sampled_index]

    def sample_sentence(self, max_length: int = 20) -> str:
        """Sample a sentence from the language model using the n-gram probabilities."""
        sentence = []
        x, y = BOS, BOS  # Start with BOS tokens as the initial context (bigram)

        for _ in range(max_length):
            # Get the probability distribution for the next word given the current context (x, y)
            next_word_probs = self.get_next_word_probs(x, y)

            # Sample the next word from the distribution using torch.multinomial
            next_word = self.sample_from_distribution(next_word_probs)

            if next_word == EOS:
                break  # Stop when we reach EOS
            
            sentence.append(next_word)  # Append sampled word to the sentence

            # Shift the context to include the last two tokens (bigram context)
            x, y = y, next_word

        # Truncate if we hit max_length, and append "..." to indicate truncation
        if len(sentence) >= max_length:
            sentence = sentence[:max_length] + ["..."]

        return " ".join(sentence)

##### SPECIFIC FAMILIES OF LANGUAGE MODELS

class CountBasedLanguageModel(LanguageModel):

    def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        # For count-based language models, it is usually convenient
        # to compute the probability first (by dividing counts) and
        # then taking the log.
        prob = self.prob(x, y, z)
        if prob == 0.0:
            return -math.inf
        return math.log(prob)

    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """Computes a smoothed estimate of the trigram probability p(z | x,y)
        according to the language model.
        """
        class_name = type(self).__name__
        if class_name == CountBasedLanguageModel.__name__:
            raise NotImplementedError("You shouldn't be calling prob on an instance of CountBasedLanguageModel, but on an instance of one of its subclasses.")
        raise NotImplementedError(
            f"{class_name}.prob is not implemented yet (you should override CountBasedLanguageModel.prob)"
        )

class UniformLanguageModel(CountBasedLanguageModel):
    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        return 1 / self.vocab_size


class AddLambdaLanguageModel(CountBasedLanguageModel):
    def __init__(self, vocab: Vocab, lambda_: float) -> None:
        super().__init__(vocab)
        if lambda_ < 0.0:
            raise ValueError(f"Negative lambda argument of {lambda_} could result in negative smoothed probs")
        self.lambda_ = lambda_

    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        assert self.event_count[x, y, z] <= self.context_count[x, y]
        return ((self.event_count[x, y, z] + self.lambda_) /
                (self.context_count[x, y] + self.lambda_ * self.vocab_size))

        # Notice that summing the numerator over all values of typeZ
        # will give the denominator.  Therefore, summing up the quotient
        # over all values of typeZ will give 1, so sum_z p(z | ...) = 1
        # as is required for any probability function.


class BackoffAddLambdaLanguageModel(AddLambdaLanguageModel):
    def __init__(self, vocab: Vocab, lambda_: float) -> None:
        super().__init__(vocab, lambda_)
        if lambda_ < 0.0:
            raise ValueError(f"Negative lambda argument of {lambda_} could result in negative smoothed probs")
        self.lambda_ = lambda_

    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        # Don't forget the difference between the Wordtype z and the
        # 1-element tuple (z,). If you're looking up counts,
        # these will have very different counts!
        p_z = (self.event_count[(z,)] + self.lambda_)/(self.context_count[()] + self.lambda_*self.vocab_size)
        p_zy = (self.event_count[(y, z)] + self.lambda_*self.vocab_size*p_z)/(self.context_count[(y,)] + self.lambda_*self.vocab_size)
        return (self.event_count[(x, y, z)] + self.lambda_*self.vocab_size*p_zy)/(self.context_count[(x, y)] + self.lambda_*self.vocab_size)


class EmbeddingLogLinearLanguageModel(LanguageModel, nn.Module):
    def __init__(self, vocab: Vocab, lexicon_file: Path, l2: float, epochs: int) -> None:
        super().__init__(vocab)
        self.l2: float = l2
        self.epochs = epochs
        
        # Initialize word-to-index mappings and embeddings
        self.word_to_idx = {word: i for i, word in enumerate(vocab)}
        self.embeddings, self.words_to_idx = self._load_embeddings(lexicon_file)
        
        # Initialize parameter matrices X, Y
        self.dim = self.embeddings.shape[1]
        self.X = nn.Parameter(torch.zeros((self.dim, self.dim)), requires_grad=True)
        self.Y = nn.Parameter(torch.zeros((self.dim, self.dim)), requires_grad=True)

        # Precompute embeddings for all vocabulary words (including OOV)
        self.z_embeds = torch.stack([self.get_embedding(word) for word in vocab])
        self.history = deque(maxlen=10)  # Keep the last 10 words by default


    def _load_embeddings(self, lexicon_file: Path) -> Tuple[torch.Tensor, Dict[str, int]]:
        words_to_idx = {}
        embeddings_lst = []
        with open(lexicon_file) as f:
            next(f)  # Skip header
            for i, line in enumerate(f):
                parts = line.split()
                word, embedding = parts[0], list(map(float, parts[1:]))
                words_to_idx[word] = i
                embeddings_lst.append(embedding)
        embeddings = torch.tensor(embeddings_lst, dtype=torch.float32)
        return embeddings, words_to_idx

    def get_embedding(self, word: str) -> torch.Tensor:
        idx = self.words_to_idx.get(word, self.words_to_idx.get("OOL"))
        return self.embeddings[idx]

    def log_prob_tensor(self, x: Wordtype, y: Wordtype, z: Wordtype) -> TorchScalar:
        logits = self.logits(x, y)  # Compute unnormalized logits
        z_index = self.word_to_idx.get(z, self.word_to_idx.get("OOL"))  # Get index of z in vocab
        log_prob = logits[z_index] - torch.logsumexp(logits, dim=0)  # Compute log p(z | xy)
        return log_prob

    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        prob = torch.exp(self.log_prob_tensor(x, y, z)).item()
        return prob
    
    def log_prob(self, x: str, y: str, z: str) -> float:
        return self.log_prob_tensor(x, y, z).item()

    def logits(self, x: Wordtype, y: Wordtype) -> Float[torch.Tensor, "vocab"]:
        x_embed = self.get_embedding(x)
        y_embed = self.get_embedding(y)
        
        # Compute logits using precomputed z_embeds
        logits = (x_embed @ self.X @ self.z_embeds.T) + (y_embed @ self.Y @ self.z_embeds.T)
        return logits

    def train(self, file: Path):
        # Initialize optimizer
        gamma0 = 1e-4 
        optimizer = optim.SGD(self.parameters(), lr=gamma0)
        nn.init.zeros_(self.X)
        nn.init.zeros_(self.Y)

        N = num_tokens(file)
        log.info(f"Start optimizing on {N} training tokens...")

        for epoch in range(self.epochs):
            total_loss = 0.0

            # Use batch training to accumulate gradients before optimizer step
            optimizer.zero_grad()

            for i, trigram in enumerate(tqdm(read_trigrams(file, self.vocab), total=N, desc=f'Epoch {epoch+1}/{self.epochs}')):
                x, y, z = trigram
                log_prob = self.log_prob_tensor(x, y, z)
                
                # Compute L2 regularization term
                l2_reg = (self.l2 / N) * (torch.norm(self.X) ** 2 + torch.norm(self.Y) ** 2)
                loss = log_prob - l2_reg  # Use negative log-prob since we want to minimize
                
                # Accumulate loss and perform backpropagation
                (-loss).backward()                
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()

            log.info(f"Epoch {epoch+1}/{self.epochs} - F: {total_loss / N }")

        log.info("done optimizing.")
        return self.X, self.Y

class ImprovedLogLinearLanguageModel(EmbeddingLogLinearLanguageModel):
    # TODO: IMPLEMENT ME!
    
    # This is where you get to come up with some features of your own, as
    # described in the reading handout.  This class inherits from
    # EmbeddingLogLinearLanguageModel and you can override anything, such as
    # `log_prob`.

    # OTHER OPTIONAL IMPROVEMENTS: You could override the `train` method.
    # Instead of using 10 epochs, try "improving the SGD training loop" as
    # described in the reading handout.  Some possibilities:
    #
    # * You can use the `draw_trigrams_forever` function that we
    #   provided to shuffle the trigrams on each epoch.
    #
    # * You can choose to compute F_i using a mini-batch of trigrams
    #   instead of a single trigram, and try to vectorize the computation
    #   over the mini-batch.
    #
    # * Instead of running for exactly 10*N trigrams, you can implement
    #   early stopping by giving the `train` method access to dev data.
    #   This will run for as long as continued training is helpful,
    #   so it might run for more or fewer than 10*N trigrams.
    #
    # * You could use a different optimization algorithm instead of SGD, such
    #   as `torch.optim.Adam` (https://pytorch.org/docs/stable/optim.html).
    #
    def __init__(self, vocab: Vocab, lexicon_file: Path, l2: float, epochs: int) -> None:
        super().__init__(vocab, lexicon_file, l2, epochs)
        
        # Initialize OOV feature vectors and weights
        self.oov_X = nn.Parameter(torch.zeros(self.dim), requires_grad=True)
        self.oov_Y = nn.Parameter(torch.zeros(self.dim), requires_grad=True)
        
        # Unigram count for frequency-based features
        self.unigram_counts = Counter()

        # Spelling-based features
        self.spelling_features = ["ing", "ed", "s"]
        self.spelling_weights = nn.Parameter(torch.zeros(len(self.spelling_features)), requires_grad=True)
        # A deque to store the recent history of words (with a fixed size)
        self.history = deque(maxlen=10)  # Keep the last 10 words by default
    
    def extract_spelling_features(self, word: Wordtype) -> torch.Tensor:
        features = torch.zeros(len(self.spelling_features))
        for i, suffix in enumerate(self.spelling_features):
            if word.endswith(suffix):
                features[i] = 1.0
        return features
    
    
    def log_prob_tensor(self, x: Wordtype, y: Wordtype, z: Wordtype) -> TorchScalar:
        logits = self.logits(x, y)
        z_index = self.word_to_idx.get(z, self.word_to_idx.get("OOL"))
        
        if z == "OOV":
            oov_feature = self.oov_X @ self.get_embedding(x) + self.oov_Y @ self.get_embedding(y)
            logits[z_index] += oov_feature

        if sum(self.unigram_counts.values()) > 0:
            unigram_prob = math.log((self.unigram_counts[z] + 1) / sum(self.unigram_counts.values()))
            logits[z_index] += unigram_prob
        
        spelling_feats = self.extract_spelling_features(z)
        logits[z_index] += self.spelling_weights @ spelling_feats
        
        if z in self.history:
            logits[z_index] += 1.0
        
        log_prob = logits[z_index] - torch.logsumexp(logits, dim=0)
        return log_prob

    def reset_history(self):
        """Clear the history queue."""
        self.history.clear()

    def train(self, file: Path, dev_file: Optional[Path] = None, patience: int = 5, batch_size: int = 32):
    # Initialize optimizer (use Adam for faster convergence)
        optimizer = torch.optim.Adam(self.parameters(), lr=2 * 1e-4)
        nn.init.zeros_(self.X)
        nn.init.zeros_(self.Y)

        # Track the best loss for early stopping
        best_loss = float('inf')
        patience_counter = 0
        self.reset_history()
        # Number of trigrams
        trigrams = list(read_trigrams(file, self.vocab))  # Load all trigrams
        N = len(trigrams)
        log.info(f"Start optimizing on {N} training trigrams...")

        for epoch in range(self.epochs):
            total_loss = 0.0

            # Shuffle the trigrams for each epoch
            np.random.shuffle(trigrams)

            # Use mini-batch training
            for i in range(0, N, batch_size):
                batch = trigrams[i:i + batch_size]
                
                # Zero the gradients
                optimizer.zero_grad()

                batch_loss = 0.0
                for trigram in batch:
                    x, y, z = trigram
                    self.history.append(z)
                    log_prob = self.log_prob_tensor(x, y, z)

                    # Compute L2 regularization term
                    l2_reg = (self.l2 / N) * (torch.norm(self.X) ** 2 + torch.norm(self.Y) ** 2)
                    loss = -log_prob + l2_reg  # Use negative log-prob since we want to minimize
                    batch_loss += loss

                # Average loss over the batch
                batch_loss = batch_loss / len(batch)

                # Backpropagation and optimization step
                batch_loss.backward()
                optimizer.step()

                total_loss += batch_loss.item()

            avg_loss = total_loss / N
            log.info(f"Epoch {epoch+1}/{self.epochs} - F: {avg_loss}")

            # Early stopping logic (optional)
            if dev_file:
                dev_loss = self.evaluate(dev_file)
                if dev_loss < best_loss:
                    best_loss = dev_loss
                    patience_counter = 0
                    torch.save(self.state_dict(), 'best_model.pth')  # Save the best model
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        log.info("Early stopping triggered!")
                        break

        log.info("done optimizing.")
        return self.X, self.Y
    # pass
