#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Tristan Meleshko <tmeleshk@ualberta.ca>
#
# Distributed under terms of the MIT license.

"""

"""
import random
from collections import defaultdict, deque

class Terminal:
    def __init__(self, text):
        self.data = [text]
        self.nickname = text
    def produce(self):
        return self.data[0]
    def __repr__(self):
        return self.nickname

class Terminator(Terminal):
    def __init__(self, fname, nickname='Terminator'):
        self.nickname=nickname
        with open(fname) as f:
            self.data = [s.strip() for s in f]
    def produce(self):
        """
        Returns a random terminal
        """
        return random.choice(self.data)
    def __repr__(self):
        return self.nickname

class Tree:
    def __init__(self, data, *nodes):
        self.data = data
        self.subtrees = []
        for node in nodes:
            self.add(node)
    def __getitem__(self, i):
        return self.subtrees[i]
    def leafs(self):
        """
        Returns all leaf nodes of the tree
        """
        if self.is_leaf():
            return [self.data]
        L = []
        for subtree in self.subtrees:
            L += subtree.leafs()
        return L
    def is_leaf(self):
        return len(self.subtrees) == 0

    def add(self, subtree_data):
        self.subtrees.append(Tree(subtree_data))
        return self.subtrees[-1]
    def __str__(self):
        return self._stringify()

    def _stringify(self, depth=0):
        s = '--'*depth + repr(self.data) + '\n'
        return (s + '\n'.join([subtree._stringify(depth+1) for subtree in self.subtrees])).rstrip()

class CFG:
    def __init__(self):
        self._grammar = defaultdict(list)

    def add_production(self, var, *outputs):
        self._grammar[var].append(outputs)

    def generate(self, cfactor=0.5, print_tree=False):
        """
        Generates random data from the grammar
        cfactor = convergence factor. A value of 1.0 causes
            uniform branching; a factor closer to 0.0 causes
            less seen branches to be prefered.
        """
        weights = defaultdict(lambda: 1.0)
        return self._generate(cfactor, 'S', weights, output=print_tree)

    def _generate(self, cfactor, start, weights, depth=0, output=False):
        if isinstance(start, Terminal):
            return start.produce()

        productions = list(enumerate(self._grammar[start]))
        i,choice = weighted_choice(productions,
                key=lambda p:weights[start,p[0]])
        if output: print '-'*depth, choice
        weights[start,i] *= cfactor
        s = ''
        for var in choice:
            s += self._generate(cfactor, var, weights, depth+1, output) + ' '
        return s.rstrip()


def weighted_choice(*values, **kwargs):
    """
    Returns a uniform random choice weighted by the given key.
    By default, returns a uniform random choice.
    Setting a key function allows you to make values more or
    less probable of being selected.

    key must be a function that takes a value as an input
    and produces a number.
    """
    key = kwargs.get('key', lambda x: 1.0)
    if len(values) == 1:
        values = values[0]
    if len(values) == 0:
        raise TypeError('weighted_choice expected 1 arguments, got 0')

    weights = [key(v) for v in values]
    s = sum(weights)
    r = random.random() * s
    for v,w in zip(values, weights):
        s -= w
        if r > s:
            return v
    return v[-1]


def main():
    G = CFG()
    noun = Terminator('data/nouns','noun')
    verb = Terminator('data/verbs','verb')
    depverb = Terminator('data/dependent_verbs','depverb')
    prep = Terminator('data/prepositions','prep')
    loc = Terminator('data/locations','loc')
    pron = Terminator('data/pronouns','pron')
    adj = Terminator('data/adjectives','adj')
    adv = Terminator('data/adverbs','adv')
    det = Terminator('data/determiners','det')
    conj = Terminator('data/conjunctions','conj')

    G.add_production('S', 'C')
    G.add_production('S', 'C', conj, 'C')
    G.add_production('C', 'DVP')
    G.add_production('NP', det, noun)
    G.add_production('NP', det, noun, 'LP')
    G.add_production('PP', prep, 'NP')
    G.add_production('LP', loc, 'NP')
    G.add_production('VP', verb)
    G.add_production('VP', 'DVP')
    G.add_production('DVP', verb, 'NP')
    G.add_production('DVP', depverb, 'NP', 'PP')

    for _ in range(1):
        print(G.generate(print_tree=True))

if __name__ == "__main__":
    main()
