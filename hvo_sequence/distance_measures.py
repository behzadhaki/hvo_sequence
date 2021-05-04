import numpy as np
import math
from hvo_sequence.feature_extractors import convert_groove
from hvo_sequence.drum_mappings import Groove_Toolbox_3Part_keymap, Groove_Toolbox_5Part_keymap


def cosine_similarity(hvo_seq_a, hvo_seq_b):
    assert hvo_seq_a.hvo.shape[-1] == hvo_seq_b.hvo.shape[-1], "the two sequences must have the same last dimension"
    assert len(hvo_seq_a.tempos) == 1 and len(hvo_seq_a.time_signatures) == 1, \
        "Input A Currently doesn't support multiple tempos or time_signatures"
    assert len(hvo_seq_b.tempos) == 1 and len(hvo_seq_b.time_signatures) == 1, \
        "Input B Currently doesn't support multiple tempos or time_signatures"

    # Ensure a and b have same length by Padding the shorter sequence to match the longer one
    max_len = max(hvo_seq_a.hvo.shape[0], hvo_seq_b.hvo.shape[0])
    shape = max_len*hvo_seq_a.hvo.shape[-1]     # Flattened shape

    a = np.zeros(shape)
    b = np.zeros(shape)

    a[:(hvo_seq_a.hvo.shape[0]*hvo_seq_a.hvo.shape[1])] = hvo_seq_a.hvo.flatten()
    b[:hvo_seq_b.hvo.shape[0]*hvo_seq_b.hvo.shape[1]] = hvo_seq_b.hvo.flatten()

    return 1-np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


def cosine_distance(hvo_seq_a, hvo_seq_b):
    return 1-cosine_similarity(hvo_seq_a, hvo_seq_b)


#######################################################################
#        Similarity Metrics from GrooveToolbox
#
#        The following code is mostly from the GrooveToolbox
#              https://github.com/fredbru/GrooveToolbox
#        Some additional functions have been implemented here
#         to adapt hvo_sequence representation to the groove and
#            utiming representations used in the GrooveToolbox
#
# Reference:    Yang, Li-Chia, and Alexander Lerch. "On the evaluation
#               of generative models in music." Neural Computing
#               and Applications 32.9 (2020): 4773-4784.
#######################################################################

def weighted_Hamming_distance(grooveA, grooveB, beat_weighting=False):

    a = grooveA.groove_all_parts
    b = grooveB.groove_all_parts

    if beat_weighting is True:
        a = _weight_groove(a)
        b = _weight_groove(b)

    x = (a.flatten()-b.flatten())
    return math.sqrt(np.dot(x, x.T))


def fuzzy_Hamming_distance(grooveA, grooveB, beat_weighting=False):
    # Get fuzzy Hamming distance as velocity weighted Hamming distance, but with 1 metrical distance lookahead/back
    # and microtiming weighting
    #

    assert grooveA.groove_all_parts.shape[0] == 32 and \
           grooveB.groove_all_parts.shape[0] == 32, "Currently only supports calculation on 2 bar " \
                                                    "loops in 4/4 and 16th note quantization"
    a = grooveA.groove_all_parts
    a_timing = grooveA.timing_matrix
    b = grooveB.groove_all_parts
    b_timing = grooveB.timing_matrix

    if beat_weighting == True:
        a = _weight_groove(a)
        b = _weight_groove(b)

    timing_difference = np.nan_to_num(a_timing - b_timing)

    x = np.zeros(a.shape)
    tempo = 120.0
    steptime_ms = 60.0 * 1000 / tempo / 4 # semiquaver step time in ms

    difference_weight = timing_difference / 125.
    difference_weight = 1+np.absolute(difference_weight)
    single_difference_weight = 400

    for j in range(a.shape[-1]):
        for i in range(31):
            if a[i,j] != 0.0 and b[i,j] != 0.0:
                x[i,j] = (a[i,j] - b[i,j]) * (difference_weight[i,j])
            elif a[i,j] != 0.0 and b[i,j] == 0.0:
                if b[(i+1) % 32, j] != 0.0 and a[(i+1) % 32, j] == 0.0:
                    single_difference = np.nan_to_num(a_timing[i,j]) - np.nan_to_num(b_timing[(i+1)%32,j]) + steptime_ms
                    if single_difference < 125.:
                        single_difference_weight = 1 + abs(single_difference_weight/steptime_ms)
                        x[i,j] = (a[i,j] - b[(i+1)%32,j]) * single_difference_weight
                    else:
                        x[i, j] = (a[i, j] - b[i, j]) * difference_weight[i, j]

                elif b[(i-1)%32,j] != 0.0 and a[(i-1)%32, j] == 0.0:
                    single_difference =  np.nan_to_num(a_timing[i,j]) - np.nan_to_num(b_timing[(i-1)%32,j]) - steptime_ms

                    if single_difference > -125.:
                        single_difference_weight = 1 + abs(single_difference_weight/steptime_ms)
                        x[i,j] = (a[i,j] - b[(i-1)%32,j]) * single_difference_weight
                    else:
                        x[i, j] = (a[i, j] - b[i, j]) * difference_weight[i, j]
                else:
                    x[i, j] = (a[i, j] - b[i, j]) * difference_weight[i, j]

            elif a[i,j] == 0.0 and b[i,j] != 0.0:
                if b[(i + 1) % 32, j] != 0.0 and a[(i + 1) % 32, j] == 0.0:
                    single_difference =  np.nan_to_num(a_timing[i,j]) - np.nan_to_num(b_timing[(i+1)%32,j]) + steptime_ms
                    if single_difference < 125.:
                        single_difference_weight = 1 + abs(single_difference_weight/steptime_ms)
                        x[i,j] = (a[i,j] - b[(i+1)%32,j]) * single_difference_weight
                    else:
                        x[i, j] = (a[i, j] - b[i, j]) * difference_weight[i, j]

                elif b[(i-1)%32,j] != 0.0 and a[(i-1)%32, j] == 0.0:
                    single_difference =  np.nan_to_num(a_timing[i,j]) - np.nan_to_num(b_timing[(i-1)%32,j]) - steptime_ms
                    if single_difference > -125.:
                        single_difference_weight = 1 + abs(single_difference_weight/steptime_ms)
                        x[i,j] = (a[i,j] - b[(i-1)%32,j]) * single_difference_weight

                    else:
                        x[i, j] = (a[i, j] - b[i, j]) * difference_weight[i, j]

                else: # if no nearby onsets, need to count difference between onset and 0 value.
                    x[i, j] = (a[i, j] - b[i, j]) * difference_weight[i, j]

        fuzzy_distance = math.sqrt(np.dot(x.flatten(), x.flatten().T))
    return fuzzy_distance


def structural_similarity_distance(grooveA, grooveB):
    # Similarity calculated between reduced versions of loops, measuring whether onsets occur in
    # roughly similar parts of two loops. Calculated as hamming distance between reduced versions.
    # of grooves
    a = grooveA.reduce_groove()
    b = grooveB.reduce_groove()
    x = (a.flatten()-b.flatten())
    structural_difference = math.sqrt(np.dot(x, x.T))
    return structural_difference


def _weight_groove(groove):
    # Metrical awareness profile weighting for hamming distance.
    # The rhythms in each beat of a bar have different significance based on GTTM.

    beat_awareness_weighting = [1, 1, 1, 1,
                                0.27, 0.27, 0.27, 0.27,
                                0.22, 0.22, 0.22, 0.22,
                                0.16, 0.16, 0.16, 0.16,
                                1, 1, 1, 1,
                                0.27, 0.27, 0.27, 0.27,
                                0.22, 0.22, 0.22, 0.22,
                                0.16, 0.16, 0.16, 0.16,]

    for i in range(groove.shape[1]):
        groove[:,i] = groove[:,i] * beat_awareness_weighting
    return groove
