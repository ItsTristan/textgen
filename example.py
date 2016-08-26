#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Tristan Meleshko <tmeleshk@ualberta.ca>
#
# Distributed under terms of the MIT license.

"""
Example showing how to generate text using textgen
"""
from __future__ import print_function
import textgen
from textgen import Terminator
from collections import defaultdict

def train_from_data(bigrams, fname, scale=1):
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
                bigrams[context, word.lower()] += scale
                context = word.lower()

def main():
    # Load files for frequency training
    from nltk.corpus import abc
    training_data = [
            'test',
            'training/big.txt',      # http://norvig.com/big.txt
            'training/extra.txt',
            '/usr/share/nltk_data/corpora/abc/science.txt',
            '/usr/share/nltk_data/corpora/abc/rural.txt'
            ]

    freq = defaultdict(lambda: 0)
    for dataset in training_data:
        print("Training on", dataset)
        try:
            train_from_data(freq, dataset)
        except IOError as e:
            print(e)


    print("=== Generated Text ===")
    G = textgen.CFG()
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

    # Epsilon production
    eps = textgen.Epsilon

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

    def weight(s, x):
        return freq[s[-1],x] - sum(w==x for w in x) + 1

    # Output
    print("\n== Sample Tree ==")

    tree = G.generate(0.5, terminals=False)
    textgen.print_tree(tree)

    print("\n== Generating using a template ==")
    print('\t', textgen.flatten_tree(tree),'\n')
    for i in range(10):
        print(textgen.to_sentence(tree, key=weight))

    print("\n== Generating a batch of samples ==")
    for tree in G.generate_batch(0.5, 10, terminals=False):
        print(textgen.to_sentence(tree, key=weight))

if __name__ == "__main__":
    main()

