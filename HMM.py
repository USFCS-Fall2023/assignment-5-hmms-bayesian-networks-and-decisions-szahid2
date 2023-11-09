# Sobia Zahid

import random
import argparse
import codecs
import os
import numpy
import numpy as np


# observations
class Observation:
    def __init__(self, stateseq, outputseq):
        self.stateseq = stateseq  # sequence of states
        self.outputseq = outputseq  # sequence of outputs

    def __str__(self):
        return ' '.join(self.stateseq) + '\n' + ' '.join(self.outputseq) + '\n'

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.outputseq)


# hmm model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities"""
        # Both of these are dictionaries of dictionaries. e.g. :
        # {'#': {'C': 0.814506898514, 'V': 0.185493101486},
        #  'C': {'C': 0.625840873591, 'V': 0.374159126409},
        #  'V': {'C': 0.603126993184, 'V': 0.396873006816}}

        self.transitions = transitions
        self.emissions = emissions

    # part 1
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""
        transition_file = "C:/Users/shyer/Downloads/two_english.trans"
        emission_file = "C:/Users/shyer/Downloads/two_english.emit"
        with open(transition_file, 'r') as trans_file:
            self.transitions = {}  # dictionary to store transition probs
            for line in trans_file:
                if not line.startswith("#"):
                    tokens = line.strip().split()
                    curr_state = tokens[0]
                    state_to = tokens[1]
                    prob = float(tokens[2])
                    if curr_state not in self.transitions:
                        self.transitions[curr_state] = {}
                    self.transitions[curr_state][state_to] = prob

        with open(emission_file, "r") as emit:
            self.emissions = {}  # dictionary to store emission probs
            for line in emit:
                tokens = line.strip().split()
                e_state = tokens[0]  # current state
                emission = tokens[1]
                eprob = float(tokens[2])
                if e_state not in self.emissions:
                    self.emissions[e_state] = {}
                self.emissions[e_state][emission] = eprob

    def generate(self, n):
        """return an n-length observation by randomly sampling from this HMM."""
        if not self.transitions or not self.emissions:
            raise ValueError("HMM model not loaded w/ probabilities.")
        observation = []
        current_state = '#' if '#' in self.transitions else random.choice(list(self.transitions.keys()))
        for _ in range(n):
            next_state = random.choices(list(self.transitions[current_state].keys()),
                                        weights=self.transitions[current_state].values())[0]
            emission = random.choices(list(self.emissions[next_state].keys()),
                                      weights=self.emissions[next_state].values())[0]
            observation.append(emission)
            current_state = next_state
        return observation

    def forward(self, observations):
        if not self.transitions or not self.emissions:
            raise ValueError("HMM model not loaded w/ probabilities.")
        n = len(observations)
        states = list(self.transitions.keys())
        num_states = len(states)
        forward_probs = np.zeros((n, num_states))
        forward_probs[0, :] = [self.emissions[state].get(observations[0], 0) for state in states]
        for t in range(1, n):
            for j, state_to in enumerate(states):
                prob_sum = sum(
                    forward_probs[t - 1, i] * self.transitions[states[i]].get(state_to, 0) for i in range(num_states))
                forward_probs[t, j] = self.emissions[state_to].get(observations[t], 0) * prob_sum
        final_state_probs = [forward_probs[n - 1, i] for i in range(num_states)]
        most_likely_final_state = states[np.argmax(final_state_probs)]
        return most_likely_final_state

    def viterbi(self, observation):
        """given an observation,
        find and return the state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """
        n = len(observations)
        states = list(self.transitions.keys())
        num_states = len(states)
        viterbi_table = np.zeros((n, num_states), dtype=float)
        bp = np.zeros((n, num_states), dtype=int)
        viterbi_table[0, :] = [self.emissions[state].get(observations[0], 0) for state in states]
        for t in range(1, n):
            for j, state_to in enumerate(states):
                max_prob = 0
                best_state = 0
                for i, state_from in enumerate(states):
                    prob = viterbi_table[t - 1, i] * self.transitions[state_from].get(state_to, 0)
                    if prob > max_prob:
                        max_prob = prob
                        best_state = i
                viterbi_table[t, j] = self.emissions[state_to].get(observations[t], 0) * max_prob
                bp[t, j] = best_state

        best_path = [0] * n
        best_path[n - 1] = np.argmax(viterbi_table[n - 1, :])
        for t in range(n - 2, -1, -1):
            best_path[t] = bp[t + 1, best_path[t + 1]]
        best_state_sequence = [states[state_index] for state_index in best_path]
        return best_state_sequence


model = HMM()
model.load('two_english')
# random_observation = model.generate(20)
# print(' '.join(random_observation))
observations = ['DET', 'NOUN', 'VERB', 'ADJ', 'NOUN']
best_state_sequence = model.viterbi(observations)
print(' '.join(best_state_sequence))
