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

class CFG:
    """
    Represents a context-free grammar
    """
    def __init__(self):
        self._grammar = defaultdict(list)

    def add_production(self, var, *outputs):
        """
        Add a prodution rule to the CFG.
        You must specify the variable 'S' at least once
        to generate text
        :var: The variable to produce from
        :outputs: The values to replace var with
        """
        self._grammar[var].append(outputs)

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
        if output: print '-'*depth, choice
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

    weights = [key(v) for v in values]
    s = sum(weights)
    r = random.random() * s
    for v,w in zip(values, weights):
        s -= w
        if r > s:
            return v
    return values[-1]

def train_from_data(bigrams, fname):
    """
    Trains the bigram table using the given file
    :bigram: An defaultdict object that stores frequencies
    :fname: The file to train from
    """
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
    # Load files for frequency training
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
    # Nouns
    noun = Terminator('data/nouns/nouns','noun')
    pnoun = Terminator('data/nouns/plurals','plural')
    pro = Terminator('data/nouns/obj_pronouns','pron')
    ppro = Terminator('data/nouns/obj_plural_pronouns', 'pron')
    cont = Terminator('data/nouns/container', 'noun')
    pcont = Terminator('data/nouns/containers', 'noun')
    snoun = Terminator('data/nouns/small_objects', 'noun')
    mnoun = Terminator('data/nouns/movable_objects', 'noun')

    # Determiners
    det = Terminator('data/determiners/only','det')
    pdet = Terminator('data/determiners/multiple','det')
    sdet = Terminator('data/determiners/singular','det')

    # Verbs
    verb = Terminator('data/verbs/verbs','verb')
    depverb = Terminator('data/verbs/motion_verbs','depverb')
    cverb = Terminator('data/verbs/container_verbs','depverb')
    sverb = Terminator('data/verbs/small_verbs','verb')

    # Prepositions
    prep = Terminator('data/prepositions/prepositions','prep')
    loc = Terminator('data/prepositions/locations','loc')
    into = Terminator('data/prepositions/contained','loc')

    # Describers
    adj = Terminator('data/adjectives/adjectives','adj')
    adv = Terminator('data/adverbs/adverbs','adv')

    # Conjunctions
    conj = Terminator('data/joins/compound','conj')

    # Grammar rules to produce from
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

    # Output
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




