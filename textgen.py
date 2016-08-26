#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Tristan Meleshko <tmeleshk@ualberta.ca>
#
# Distributed under terms of the MIT license.

"""
Text generation in python using CFG
"""
from __future__ import print_function
import random
from collections import defaultdict, deque

class Terminal:
    """
    Represents a terminal in a CFG.
    Use this as the base class for any Terminal objects
    you define.
    """
    def __init__(self, *text, **kwargs):
        self.data = list(text)
        self.nickname = kwargs.get('nickname', 'Terminal')

    def produce(self, key=lambda x: 1.0):
        """
        Produces the best item from the terminal's dataset
        given the key
        """
        return max(self.data[0], key=key)

    def __repr__(self):
        return self.nickname

class Terminator(Terminal):
    """
    Generates a random word using a provided text file
    as its data source.
    Each line in the text file is treated as a separate terminal.
    """
    def __init__(self, fname, nickname='Terminator', cfactor=0.8):
        """
        :fname: The filename containing the text to produce from.
        :nickname: The text to display to represent this terminal variable
        :cfactor: The amount to the reduce the probability of selecting a word
        """
        self.nickname=nickname
        self.cfactor=cfactor
        self._weights = defaultdict(lambda: 1.0)
        with open(fname) as f:
            self.data = [s.strip() for s in f]

    def produce(self,key=lambda x: 1.0):
        """
        Returns a random terminal from the dataset as a
        weighted choice given the key.
        """
        i = weighted_choice(range(len(self.data)), key=lambda i: key(self.data[i]))
        self._update(i)
        return self.data[i]

    def reset(self):
        """
        Reset the internal state of the object
        """
        self._weights.clear()

    def _update(self, index):
        self._weights[index] *= self.cfactor

    def __repr__(self):
        return self.nickname

Epsilon = Terminal('', nickname='eps')

class CFG:
    """
    Represents a context-free grammar
    """
    def __init__(self):
        self._grammar = defaultdict(list)
        self._terminals = set()

    def add_production(self, var, *outputs):
        """
        Add a prodution rule to the CFG.
        You must specify the variable 'S' at least once
        to generate text
        :var: The variable to produce from
        :outputs: The values to replace var with
        """
        self._grammar[var].append(outputs)
        self._terminals.add(o for o in outputs if isinstance(o, Terminal))

    def generate_batch(self, cfactor, nsamples, terminals=True):
        """
        Generate a random batch of sample text
        :cfactor: The amount to reduce the probability of selection.
            A factor of 1.0 is naive recursive generation
            This value must be strictly positive (>0)
        :nsamples: Number of samples to produce
        :terminals: Convert terminal objects to text. (Default True)
        """
        weights = defaultdict(lambda: 1.0)
        return [self._generate(cfactor,'S',weights, terms=terminals) for _ in range(nsamples)]

    def generate(self, cfactor=0.5, print_tree=False, terminals=True):
        """
        Generates random data from the grammar
        :cfactor: The amount to reduce the probability of selection.
            A factor of 1.0 is naive recursive generation
            This value must be strictly positive (>0)
        :print_tree: = print tree for debugging
        :terminals: Convert terminal objects to text. (default True)
        """
        weights = defaultdict(lambda: 1.0)
        return self._generate(cfactor, 'S', weights, output=print_tree, terms=terminals)

    def _generate(self, cfactor, start, weights={}, depth=0, output=False, terms=True):
        if isinstance(start, Terminal):
            if terms:
                return start, start.produce()
            else:
                return start, start

        productions = list(enumerate(self._grammar[start]))
        i,choice = weighted_choice(productions,
                key=lambda p:weights[start,p[0]])
        if output: print('-'*depth, choice)
        weights[start,i] *= cfactor
        s = []
        for var in choice:
            s.append(self._generate(cfactor, var, weights, depth+1, output, terms))
        return start, s

def to_sentence(tree, key=lambda s,x: 1.0):
    """
    Convert a tree object into a string, using key to produce
    values from Terminals
    :tree: The tree constructed by CFG.generate()
    :key: A function mapping the sentence so far (from left-to-right)
        and a word to a probability of selection.
        Allows you to weigh word choices (x) based on context.
        Example:
            If you have a table freq that maps from a 2-tuple of words
            to a frequency of usage, then you can use a key of
                key = lambda s,x: freq[s[-1], x]
            to produce a word using the previous word in the sentence.
            Note that s is fixed but x is iterated over every word
    """
    s = ['']
    for terminal in flatten_tree(tree):
        if isinstance(terminal, str):
            s.append(terminal)
        else:
            s.append(terminal.produce(key=lambda x: key(s,x)))
    s.pop(0)
    return ' '.join(s)

def flatten_tree(tree):
    """
    Given a tree, produces only its leaf nodes
    :tree: The tree constructed by CFG.generate()
    """
    if isinstance(tree[1],str):
        return [tree[1]]
    if isinstance(tree[1],Terminal):
        return [tree[1]]
    s = []
    for subtree in tree[1]:
        s += flatten_tree(subtree)
    return s

def print_tree(tree, depth=0):
    """
    Print the given tree and its structure
    :tree: The tree constructed by CFG.generate()
    """
    print('+','--'*depth,tree[0])
    if isinstance(tree[1], str):
        print('|','  '*depth,'->',tree[1])
        return
    if isinstance(tree[1],Terminal):
        print('|','  '*depth,'->',repr(tree[1]))
        return
    for subtree in tree[1]:
        print_tree(subtree, depth+1)

def weighted_choice(*values, **kwargs):
    """
    Returns a uniform random choice weighted by the given key.
    By default, returns a uniform random choice.
    Setting a key function allows you to make values more or
    less probable of being selected.

    key must be a function that takes a value as an input
    and produces a number.

    Valid Usage:
        weighted_choice(x1, x2, ..., xn)
        weighted_choice([x1, x2, ... xn])

    :values: An iterator of values to choose from
    :key: A function mapping from a value to a number to weigh
        the probabilities
    """
    key = kwargs.get('key', lambda x: 1.0)
    if len(values) == 1:
        values = values[0]
    if len(values) == 0:
        raise TypeError('weighted_choice expected 1 arguments, got 0')

    weights = {v:key(v) for v in values}
    total = sum(weights.values())
    r = random.uniform(0, total)
    upto = 0
    for v, w in weights.items():
        if upto + w >= r:
            return v
        upto += w
    assert False


