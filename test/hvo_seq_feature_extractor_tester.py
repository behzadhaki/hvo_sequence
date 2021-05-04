from hvo_sequence.hvo_seq import HVO_Sequence
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING, Groove_Toolbox_5Part_keymap, Groove_Toolbox_3Part_keymap
from hvo_sequence.feature_extractors import GrooveToolbox, CompleteFeatureExtractor

import numpy as np


if __name__ == "__main__":

    # Create an instance of a HVO_Sequence
    hvo_seq = HVO_Sequence(drum_mapping=ROLAND_REDUCED_MAPPING)
    hvo_seq_b = HVO_Sequence(drum_mapping=ROLAND_REDUCED_MAPPING)

    # Add a time_signature
    hvo_seq.add_time_signature(0, 4, 4, [4])
    hvo_seq_b.add_time_signature(0, 4, 4, [4])

    # Add two tempos
    hvo_seq.add_tempo(0, 50)
    hvo_seq_b.add_tempo(0, 50)

    # hvo_seq.add_tempo(12, 20)  # Tempo Change at the beginning of second bar

    # Create a random hvo
    hits = np.random.randint(0, 2, (16, 9))
    vels = hits * np.random.rand(16, 9)
    offs = hits * (np.random.rand(16, 9) - 0.5)

    vels_b = hits * np.random.rand(16, 9)
    offs_b = hits * (np.random.rand(16, 9) - 0.5)

    # Add hvo score to hvo_seq instance
    hvo_seq.hvo = np.concatenate((hits, vels, offs), axis=1)
    hvo_seq_b.hvo = np.concatenate((hits, vels_b, offs_b), axis=1)

    # Calculate distance metrics
    print(hvo_seq.calculate_all_distances_with(hvo_seq_b))
    print(hvo_seq.calculate_all_distances_with(hvo_seq))


    """groove = GrooveToolbox(hvo_seq,_5kitparts_map=Groove_Toolbox_5Part_keymap,
                           _3kitparts_map=Groove_Toolbox_3Part_keymap)"""

    """groove.RhythmFeatures.calculate_all_features()
    print("--"*100)
    groove.MicrotimingFeatures.print_all_features()
    """
    """
    FeatureExtractor = CompleteFeatureExtractor(hvo_seq, _5kitparts_map=Groove_Toolbox_5Part_keymap,
                                                _3kitparts_map=Groove_Toolbox_3Part_keymap)

    print(FeatureExtractor.get_feature_dictionary())"""