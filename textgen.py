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
    def __init__(self, fname, nickname='Terminator', cfactor=.8):
        self.nickname=nickname
        self.cfactor=cfactor
        self._weights = defaultdict(lambda: 1.0)
        with open(fname) as f:
            self.data = [s.strip() for s in f]
    def produce(self):
        """
        Returns a random terminal
        """
        i = weighted_choice(range(len(self.data)), key=self._key)
        self._update(i)
        return self.data[i]
    def _update(self, index):
        self._weights[index] *= self.cfactor
    def _key(self, index):
        return self._weights[index]
    def reset(self):
        self._weights.clear()
    def __repr__(self):
        return self.nickname

class CFG:
    def __init__(self):
        self._grammar = defaultdict(list)

    def add_production(self, var, *outputs):
        self._grammar[var].append(outputs)

    def generate_batch(self, cfactor, nsamples):
        weights = defaultdict(lambda: 1.0)
        return [self._generate(cfactor,'S',weights) for _ in range(nsamples)]

    def generate(self, cfactor=0.5, print_tree=False):
        """
        Generates random data from the grammar
        cfactor = convergence factor. A value of 1.0 causes
            uniform branching; a factor closer to 0.0 causes
            less seen branches to be prefered.
        """
        weights = defaultdict(lambda: 1.0)
        return self._generate(cfactor, 'S', weights, output=print_tree)

    def _generate(self, cfactor, start, weights={}, depth=0, output=False):
        if isinstance(start, Terminal):
            return start, start.produce()

        productions = list(enumerate(self._grammar[start]))
        i,choice = weighted_choice(productions,
                key=lambda p:weights[start,p[0]])
        if output: print '-'*depth, choice
        weights[start,i] *= cfactor
        s = []
        for var in choice:
            s.append(self._generate(cfactor, var, weights, depth+1, output))
        return start, s

def flatten_tree(tree):
    if isinstance(tree[1],str):
        return [tree[1]]
    s = []
    for subtree in tree[1]:
        s += flatten_tree(subtree)
    return s
def print_tree(tree, depth=0):
    print '+','--'*depth,tree[0]
    if isinstance(tree[1], str):
        print '|','  '*depth,'->',tree[1]
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
    noun = Terminator('data/nouns/nouns','noun')
    pnoun = Terminator('data/nouns/plurals','plural')
    pro = Terminator('data/nouns/obj_pronouns','pron')
    ppro = Terminator('data/nouns/obj_plural_pronouns', 'pron')
    cont = Terminator('data/nouns/container', 'noun')
    pcont = Terminator('data/nouns/containers', 'noun')

    det = Terminator('data/determiners/only','det')
    pdet = Terminator('data/determiners/multiple','det')
    sdet = Terminator('data/determiners/singular','det')

    verb = Terminator('data/verbs/verbs','verb')
    depverb = Terminator('data/verbs/dependent_verbs','depverb')

    prep = Terminator('data/prepositions/prepositions','prep')
    loc = Terminator('data/prepositions/locations','loc')
    into = Terminator('data/prepositions/contained','loc')

    adj = Terminator('data/adjectives/adjectives','adj')
    adv = Terminator('data/adverbs/adverbs','adv')


    conj = Terminator('data/joins/compound','conj')

    G.add_production('S', 'C')
    G.add_production('S', 'C', conj, 'C')
    G.add_production('C', 'DVP')
    G.add_production('NP', det, noun)
    G.add_production('NP', pdet, pnoun)
    G.add_production('NP', sdet, noun)
    G.add_production('NP', det, noun, 'LP')
    G.add_production('NP', pdet, pnoun, 'LP')
    G.add_production('CNP', det, cont)
    G.add_production('CNP', pdet, pcont)
    G.add_production('CNP', sdet, cont)
    G.add_production('CNP', det, cont, 'LP')
    G.add_production('CNP', pdet, pcont, 'LP')
    G.add_production('LP', prep, 'NP')
    G.add_production('PP', into, 'CNP')
    G.add_production('PP', loc, 'NP')
    G.add_production('VP', verb)
    G.add_production('VP', 'DVP')
    G.add_production('DVP', verb, 'NP')
    G.add_production('DVP', depverb, 'NP', 'PP')

    for sentence in G.generate_batch(0.5, 10):
        print ' '.join(flatten_tree(sentence))
    sentence = G.generate(0.5)
    print_tree(sentence)
    print ' '.join(flatten_tree(sentence))

if __name__ == "__main__":
    main()
