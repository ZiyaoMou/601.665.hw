#!/usr/bin/env python3
"""
Determine whether sentences are grammatical under a CFG, using Earley's algorithm.
(Starting from this basic recognizer, you should write a probabilistic parser
that reconstructs the highest-probability parse of each given sentence.)
"""

# Recognizer code by Arya McCarthy, Alexandra DeLucia, Jason Eisner, 2020-10, 2021-10.
# This code is hereby released to the public domain.

from __future__ import annotations
import argparse
import copy
import logging
import math
import sys

import tqdm
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Counter as CounterType, Iterable, List, Optional, Dict, Tuple

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "grammar", type=Path, help="Path to .gr file containing a PCFG'"
    )
    parser.add_argument(
        "sentences", type=Path, help="Path to .sen file containing tokenized input sentences"
    )
    parser.add_argument(
        "-s",
        "--start_symbol",
        type=str,
        help="Start symbol of the grammar (default is ROOT)",
        default="ROOT",
    )

    parser.add_argument(
        "--progress",
        action="store_true",
        help="Display a progress bar",
        default=False,
    )

    # for verbosity of logging
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="logging_level", action="store_const", const=logging.WARNING
    )

    return parser.parse_args()


class EarleyChart:
    """A chart for Earley's algorithm."""

    def __init__(self, tokens: List[str], grammar: Grammar, progress: bool = False) -> None:
        """Create the chart based on parsing `tokens` with `grammar`.  
        `progress` says whether to display progress bars as we parse."""
        self.tokens = tokens
        self.grammar = grammar
        self.grammar.vocab_specialize(tokens)
        self.progress = progress
        self.profile: CounterType[str] = Counter()

        self.cols: List[Agenda]
        self._run_earley()  # run Earley's algorithm to construct self.cols
        self.result = ''


    def display_optimal_parse(self):
        """Determine if the sentence was accepted and print the optimal parse with the lowest weight."""
        optimal_parse = min(
            (state for state in self.cols[-1].all()
            if state.rule.lhs == self.grammar.start_symbol 
            and state.next_symbol() is None 
            and state.start_position == 0),
            key=lambda state: state.rule.weight, default=None
        )

        if optimal_parse:
            self.result = ''
            self.print_item(optimal_parse)  # Corrected to use print_item
            print(self.result.strip())
            print(str(optimal_parse.rule.weight))
        else:
            print('NONE')

    # def _merge_completed_items(self, nonterminal: str, position: int) -> None:
    #     """Merges all completed items for the nonterminal starting at the given position."""
    #     for item in self.cols[position].all():
    #         if item.rule.lhs == nonterminal and item.next_symbol() is None:
    #             self._attach(item, position)

    def _run_earley(self) -> None:
        """Fill in the Earley chart."""
        # Initially empty column for each position in sentence
        self.cols = [Agenda() for _ in range(len(self.tokens) + 1)]

        # Start looking for ROOT at position 0
        self._predict(self.grammar.start_symbol, 0)

        # We'll go column by column, and within each column row by row.
        # Processing earlier entries in the column may extend the column
        # with later entries, which will be processed as well.
        # 
        # The iterator over numbered columns is `enumerate(self.cols)`.  
        # Wrapping this iterator in the `tqdm` call provides a progress bar.
        for i, column in tqdm.tqdm(enumerate(self.cols),
                                   total=len(self.cols),
                                   disable=not self.progress):
            log.debug("")
            log.debug(f"Processing items in column {i}")
            while column:  # while agenda isn't empty
                item = column.pop()  # dequeue the next unprocessed item
                next = item.next_symbol();
                if next is None:
                    # Attach this complete constituent to its customers
                    log.debug(f"{item} => ATTACH")
                    self._attach(item, i)
                elif self.grammar.is_nonterminal(next):
                    # Predict the nonterminal after the dot
                    log.debug(f"{item} => PREDICT")
                    self._predict(next, i)
                else:
                    # Try to scan the terminal after the dot
                    log.debug(f"{item} => SCAN")
                    self._scan(item, i)

    def _predict(self, nonterminal: str, position: int) -> None:
        """Start looking for this nonterminal at the given position."""
        if self.cols[position].is_predicted(nonterminal):
            return
        
        self.cols[position].mark_predicted(nonterminal)

        for rule in self.grammar.expansions(nonterminal):
            new_item = Item(rule, dot_position=0, start_position=position)
            self.cols[position].add_to_batch(new_item)
            log.debug(f"\tPredicted: {new_item} in column {position}")
            self.profile["PREDICT"] += 1

    def _scan(self, item: Item, position: int) -> None:
        """Attach the next word to this item that ends at position, 
        if it matches what this item is looking for next."""
        if position < len(self.tokens) and self.tokens[position] == item.next_symbol():
            new_item = item.with_dot_advanced(item, self.tokens[position])
            self.cols[position + 1].push(new_item)
            log.debug(f"\tScanned to get: {new_item} in column {position + 1}")
            self.profile["SCAN"] += 1

    def _attach(self, item: Item, position: int) -> None:
        """Attach this complete item to its customers in previous columns, advancing the
        customers' dots to create new items in this column.  (This operation is sometimes
        called "complete," but actually it attaches an item that was already complete.)
        """
        mid = item.start_position  # start position of this item = end position of item to its left
        for customer in self.cols[mid].all():  # could you eliminate this inefficient linear search?
            if customer.next_symbol() == item.rule.lhs:
                new_item = customer.with_dot_advanced(customer, item, item.rule.weight)
                self.cols[position].push(new_item)
                # self._merge_completed_items(new_item)  # Merge all completed items for the nonterminal
                log.debug(f"\tAttached to get: {new_item} in column {position}")
                self.profile["ATTACH"] += 1

    def print_item(self, item):
        if isinstance(item, Item):
            if not item.next_symbol():
                # Append the current item's left-hand side (LHS) and recursively print its constituents
                self.result += f' ({item.rule.lhs}'
                self.print_item(item.parent_state)
                self.print_item(item.new_state)
                self.result += ')'
            else:
                # Recursively print the parent and new state
                self.print_item(item.parent_state)
                self.print_item(item.new_state)
        else:
            # If it's not an Item, add it directly to the result
            self.result += f' {item}'


class Agenda:
    """An agenda of items that need to be processed.  Newly built items 
    may be enqueued for processing by `push()`, and should eventually be 
    dequeued by `pop()`.

    This implementation of an agenda also remembers which items have
    been pushed before, even if they have subsequently been popped.
    This is because already popped items must still be found by
    duplicate detection and as customers for attach.  

    (In general, AI algorithms often maintain a "closed list" (or
    "chart") of items that have already been popped, in addition to
    the "open list" (or "agenda") of items that are still waiting to pop.)

    In Earley's algorithm, each end position has its own agenda -- a column
    in the parse chart.  (This contrasts with agenda-based parsing, which uses
    a single agenda for all items.)

    Standardly, each column's agenda is implemented as a FIFO queue
    with duplicate detection, and that is what is implemented here.
    However, other implementations are possible -- and could be useful
    when dealing with weights, backpointers, and optimizations.

    # >>> a = Agenda()
    # >>> a.push(3)
    # >>> a.push(5)
    # >>> a.push(3)   # duplicate ignored
    # >>> a
    # Agenda([]; [3, 5])
    # >>> a.pop()
    # 3
    # >>> a
    # Agenda([3]; [5])
    # >>> a.push(3)   # duplicate ignored
    # >>> a.push(7)
    # >>> a
    # Agenda([3]; [5, 7])
    # >>> while a:    # that is, while len(a) != 0
    # ...    print(a.pop())
    5
    7

    """

    def __init__(self) -> None:
        self._items: List[Item] = []  # list of all items that were *ever* pushed
        self._index: Dict[Item, int] = {}  # stores index of an item if it was ever pushed
        self._next = 0  # index of first item that has not yet been popped
        self._reprocess: List[int] = []  # items by index that were popped but need to be reprocessed
        self._predicted: set[str] = set()  # stores batch of predicted non-terminals

        # Note: There are other possible designs.  For example, self._index doesn't really
        # have to store the index; it could be changed from a dictionary to a set.  
        # 
        # However, we provided this design because there are multiple reasonable ways to extend
        # this design to store weights and backpointers.  That additional information could be
        # stored either in self._items or in self._index{}.

    def __len__(self) -> int:
        """Returns number of items that are still waiting to be popped.
        Enables `len(my_agenda)`."""
        return len(self._items) + len (self._reprocess) - self._next
    
    def add_to_batch(self, item: Item) -> None:
        """Add (enqueue) the item knowing the category is already in the agenda"""
        self._items.append(item)
        self._index[item] = len(self._items) - 1

    def push(self, item: Item) -> None:
        """Add (enqueue) the item, unless it was previously added."""
        if item not in self._index:  # O(1) lookup in hash table
            self._items.append(item)
            self._index[item] = len(self._items) - 1
        else:
            old_index = self._index[item]
            if item.rule.weight < self._items[old_index].rule.weight:
                self._items[old_index] = item
                if old_index not in self._reprocess:
                    self._reprocess.append(old_index)

    def mark_predicted(self, nonterminal: str) -> None:
        self._predicted.add(nonterminal)

    def is_predicted(self, nonterminal: str) -> bool:
        return nonterminal in self._predicted

    def pop(self) -> Item:
        """Returns one of the items that was waiting to be popped (dequeued).
        Raises IndexError if there are no items waiting."""
        if len(self) == 0:
            raise IndexError
        if self._reprocess:
            index = self._reprocess.pop()
            item = self._items[index]
            return item
        item = self._items[self._next]
        self._next += 1
        return item

    def all(self) -> Iterable[Item]:
        """Collection of all items that have ever been pushed, even if 
        they've already been popped."""
        return self._items

    def __repr__(self):
        """Provide a human-readable string REPResentation of this Agenda."""
        next = self._next
        return f"{self.__class__.__name__}({self._items[:next]}; {self._items[next:]})"


class Grammar:
    """Represents a weighted context-free grammar."""

    def __init__(self, start_symbol: str, *files: Path) -> None:
        """Create a grammar with the given start symbol, 
        adding rules from the specified files if any."""
        self.start_symbol = start_symbol
        self._expansions: Dict[str, List[Rule]] = {}  # maps each LHS to the list of rules that expand it
        # Read the input grammar files
        for file in files:
            self.add_rules_from_file(file)

    def add_rules_from_file(self, file: Path) -> None:
        """Add rules to this grammar from a file (one rule per line).
        Each rule is preceded by a normalized probability p,
        and we take -log2(p) to be the rule's weight."""
        with open(file, "r") as f:
            for line in f:
                # remove any comment from end of line, and any trailing whitespace
                line = line.split("#")[0].rstrip()
                # skip empty lines
                if line == "":
                    continue
                # Parse tab-delimited linfore of format <probability>\t<lhs>\t<rhs>
                _prob, lhs, _rhs = line.split("\t")
                prob = float(_prob)
                rhs = tuple(_rhs.split())
                rule = Rule(lhs=lhs, rhs=rhs, weight=-math.log2(prob))
                if lhs not in self._expansions:
                    self._expansions[lhs] = []
                self._expansions[lhs].append(rule)

    def expansions(self, lhs: str) -> Iterable[Rule]:
        """Return an iterable collection of all rules with a given lhs"""
        return self._expansions[lhs]

    def is_nonterminal(self, symbol: str) -> bool:
        """Is symbol a nonterminal symbol?"""
        return symbol in self._expansions

    def is_relevant(self, rule: Rule, tokens: List[str]) -> bool:
        """Is rule relevant to the sentence?"""
        for token in rule.rhs:
            if self.is_nonterminal(token):
                continue
            else:
                if token not in tokens:
                    return False
        return True

    def vocab_specialize(self, tokens: List[str]) -> None:
        """Specialize the vocabulary of the grammar to the given tokens."""
        specialized = {
            key: [rule for rule in rules if self.is_relevant(rule, tokens)]
            for key, rules in self._expansions.items()
        }
        if not specialized:  # Add fallback if no specialized rules match
            print("Warning: No specialized rules found. Using full grammar.")
        else:
            self._expansions = specialized


# A dataclass is a class that provides some useful defaults for you. If you define
# the data that the class should hold, it will automatically make things like an
# initializer and an equality function.  This is just a shortcut.  
# More info here: https://docs.python.org/3/library/dataclasses.html
# Using a dataclass here lets us specify that instances are "frozen" (immutable),
# and therefore can be hashed and used as keys in a dictionary.
@dataclass(frozen=True)
class Rule:
    """
    A grammar rule has a left-hand side (lhs), a right-hand side (rhs), and a weight.

    # >>> r = Rule('S',('NP','VP'),3.14)
    # >>> r
    # S → NP VP
    # >>> r.weight
    # 3.14
    # >>> r.weight = 2.718
    # Traceback (most recent call last):
    # dataclasses.FrozenInstanceError: cannot assign to field 'weight'
    """
    lhs: str
    rhs: Tuple[str, ...]
    weight: float = 0.0

    def __repr__(self) -> str:
        """Complete string used to show this rule instance at the command line"""
        # Note: You might want to modify this to include the weight.
        return f"{self.lhs} → {' '.join(self.rhs)}"

    def add_weight(self, weight):
        return Rule(lhs=self.lhs, rhs=self.rhs, weight=self.weight + weight)


# We particularly want items to be immutable, since they will be hashed and 
# used as keys in a dictionary (for duplicate detection).  
@dataclass(frozen=True)
class Item:
    """An item in the Earley parse chart, representing one or more subtrees
    that could yield a particular substring."""
    rule: Rule
    dot_position: int
    start_position: int
    parent_state: Item = None
    new_state: Item = None

    # We don't store the end_position, which corresponds to the column
    # that the item is in, although you could store it redundantly for 
    # debugging purposes if you wanted.

    def next_symbol(self) -> Optional[str]:
        """What's the next, unprocessed symbol (terminal, non-terminal, or None) in this partially matched rule?"""
        assert 0 <= self.dot_position <= len(self.rule.rhs)
        if self.dot_position == len(self.rule.rhs):
            return None
        else:
            return self.rule.rhs[self.dot_position]

    def with_dot_advanced(self, parent_state, new_state, weight=None) -> Item:
        if self.next_symbol() is None:
            raise IndexError("Can't advance the dot past the end of the rule")
        if weight:
            new_rule = self.rule.add_weight(weight)
            return Item(rule=new_rule, dot_position=self.dot_position + 1, start_position=self.start_position,
                        parent_state=parent_state, new_state=new_state)
        else:
            return Item(rule=self.rule, dot_position=self.dot_position + 1, start_position=self.start_position,
                        parent_state=parent_state, new_state=new_state)

    def __repr__(self) -> str:
        """Human-readable representation string used when printing this item."""
        # Note: If you revise this class to change what an Item stores, you'll probably want to change this method too.
        DOT = "·"
        rhs = list(self.rule.rhs)  # Make a copy.
        rhs.insert(self.dot_position, DOT)
        dotted_rule = f"{self.rule.lhs} → {' '.join(rhs)}"
        return f"({self.start_position}, {dotted_rule})"  # matches notation on slides

    def __key(self):
        return self.dot_position, self.start_position, self.rule.lhs, self.rule.rhs

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, Item):
            return self.__key() == other.__key()
        return NotImplemented


def main():
    # Parse the command-line arguments
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    grammar = Grammar(args.start_symbol, args.grammar)

    with open(args.sentences) as f:
        for sentence in f.readlines():
            sentence = sentence.strip()
            if sentence != "":  # skip blank lines
                # analyze the sentence
                log.debug("=" * 70)
                log.debug(f"Parsing sentence: {sentence}")
                chart = EarleyChart(sentence.split(), copy.deepcopy(grammar), progress=args.progress)
                chart.display_optimal_parse()
                log.debug(f"Profile of work done: {chart.profile}")


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=False)  # run tests
    main()