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

    def produce(self,key=lambda x: 1.0):
        """
        Returns a random terminal
        """
        i = weighted_choice(range(len(self.data)), key=lambda i: key(self.data[i]))
        self._update(i)
        return self.data[i]
    def _update(self, index):
        self._weights[index] *= self.cfactor
    def reset(self):
        self._weights.clear()
    def __repr__(self):
        return self.nickname

class CFG:
    def __init__(self):
        self._grammar = defaultdict(list)

    def add_production(self, var, *outputs):
        self._grammar[var].append(outputs)

    def generate_batch(self, cfactor, nsamples, terminals=True):
        weights = defaultdict(lambda: 1.0)
        return [self._generate(cfactor,'S',weights, terms=terminals) for _ in range(nsamples)]

    def generate(self, cfactor=0.5, print_tree=False, terminals=True):
        """
        Generates random data from the grammar
        cfactor = convergence factor. A value of 1.0 causes
            uniform branching; a factor closer to 0.0 causes
            less seen branches to be prefered.
        print_tree = print tree for debugging
        terminals = set to False to disable the production of terminals
            (i.e., leave them as Terminal objects)
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
        if output: print '-'*depth, choice
        weights[start,i] *= cfactor
        s = []
        for var in choice:
            s.append(self._generate(cfactor, var, weights, depth+1, output, terms))
        return start, s

def to_sentence(tree, key=lambda s,x: 1.0):
    s = ['']
    for terminal in flatten_tree(tree):
        if isinstance(terminal, str):
            s.append(terminal)
        else:
            s.append(terminal.produce(key=lambda x: key(s,x)))
    s.pop(0)
    return ' '.join(s)

def flatten_tree(tree):
    if isinstance(tree[1],str):
        return [tree[1]]
    if isinstance(tree[1],Terminal):
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
    if isinstance(tree[1],Terminal):
        print '|','  '*depth,'->',repr(tree[1])
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
    return values[-1]

def train_from_data(bigrams, fname):
    context = ''
    def remove_punct(word):
        punct = """-_,. :;"'()[]{}$!?/\\"""
        for c in punct:
            word = word.replace(c,' ')
        return word
    with open(fname) as f:
        for line in f:
            line = remove_punct(line)
            for word in line.split():
                if not word: continue
                bigrams[context, word.lower()] += 1
                context = word.lower()

def main():
    training_data = [
            'training/big.txt',      # http://norvig.com/big.txt
            'training/extra.txt'
            ]

    freq = defaultdict(lambda: 0)
    for dataset in training_data:
        print "Training on", dataset
        train_from_data(freq,dataset)

    print "=== Generated Text ==="
    G = CFG()
    noun = Terminator('data/nouns/nouns','noun')
    pnoun = Terminator('data/nouns/plurals','plural')
    pro = Terminator('data/nouns/obj_pronouns','pron')
    ppro = Terminator('data/nouns/obj_plural_pronouns', 'pron')
    cont = Terminator('data/nouns/container', 'noun')
    pcont = Terminator('data/nouns/containers', 'noun')
    snoun = Terminator('data/nouns/small_objects', 'noun')
    mnoun = Terminator('data/nouns/movable_objects', 'noun')

    det = Terminator('data/determiners/only','det')
    pdet = Terminator('data/determiners/multiple','det')
    sdet = Terminator('data/determiners/singular','det')

    verb = Terminator('data/verbs/verbs','verb')
    depverb = Terminator('data/verbs/motion_verbs','depverb')
    cverb = Terminator('data/verbs/container_verbs','depverb')
    sverb = Terminator('data/verbs/small_verbs','verb')

    prep = Terminator('data/prepositions/prepositions','prep')
    loc = Terminator('data/prepositions/locations','loc')
    into = Terminator('data/prepositions/contained','loc')

    adj = Terminator('data/adjectives/adjectives','adj')
    adv = Terminator('data/adverbs/adverbs','adv')


    conj = Terminator('data/joins/compound','conj')

    G.add_production('S', 'C')
    G.add_production('S', 'C', conj, 'C')
    G.add_production('C', 'DVP')
    G.add_production('NP', 'TNP')
    G.add_production('NP', 'TNP', 'PP')
    G.add_production('TNP', 'CNP')
    G.add_production('TNP', 'SNP')
    G.add_production('TNP', 'BNP')
    G.add_production('MNP', det, mnoun)
    G.add_production('MNP', sdet, mnoun)
    G.add_production('BNP', det, noun)
    G.add_production('BNP', pdet, pnoun)
    G.add_production('BNP', sdet, noun)
    G.add_production('SNP', det, snoun)
    G.add_production('SNP', sdet, snoun)
    G.add_production('CNP', det, cont)
    G.add_production('CNP', pdet, pcont)
    G.add_production('CNP', sdet, cont)
    G.add_production('PP', 'IPP')
    G.add_production('PP', 'LPP')
    G.add_production('IPP', into, 'CNP')
    G.add_production('LPP', loc, 'TNP')
    G.add_production('VP', verb)
    G.add_production('VP', 'DVP')
    G.add_production('VP', 'SVP')
    G.add_production('DVP', verb, 'TNP')
    G.add_production('DVP', cverb, 'CNP')
    G.add_production('DVP', depverb, 'SNP', 'IPP')
    G.add_production('DVP', depverb, 'MNP', 'LPP')
    G.add_production('SVP', sverb, 'SNP')
    G.add_production('SVP', sverb, 'SNP', 'IPP')
    G.add_production('SVP', sverb, 'SNP', 'LPP')

    print "\n== Sample Tree =="
    tree = G.generate(0.5, terminals=False)
    print_tree(tree)
    print "\n== Generating using a template =="
    print "\t", flatten_tree(tree),'\n'
    for i in xrange(10):
        print to_sentence(tree, key=lambda s,x: freq[s[-1],x])
    print "\n== Generating a batch of samples =="
    for tree in G.generate_batch(0.5, 10, terminals=False):
        print to_sentence(tree, key=lambda s,x: freq[s[-1],x])

if __name__ == "__main__":
    main()
