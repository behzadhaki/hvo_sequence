from hvo_sequence.hvo_seq import HVO_Sequence
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING

import numpy as np


if __name__ == "__main__":

    hvo_seq = HVO_Sequence()
    hvo_seq.add_tempo(0, 120)

    hvo_seq.add_time_signature(0, 4, 4, [4])

    hvo_seq.add_tempo(7, 50)      # Tempo Change at the beginning of second bar

    # hvo_seq.add_time_signature(8, 6, 8, [3])

    hvo_seq.drum_mapping = ROLAND_REDUCED_MAPPING

    hits = np.random.randint(0, 2, (36, 9))
    vels = hits * np.random.rand(36, 9)
    offs = hits * (np.random.rand(36, 9) - 0.5)
    hvo_seq.hvo = np.concatenate((hits, vels, offs), axis=1)

    step_ix = 0
    time_signature_boundaries = np.array(hvo_seq.tempo_consistent_segment_boundaries)
    time_signature_boundaries_distance = np.array(time_signature_boundaries) - step_ix

    print(hvo_seq.steps_per_beat_per_segments)
    # print(hvo_seq.grid_lines)
    # print(hvo_seq.grid_lines.shape)
    # print(hvo_seq.n_beats_per_segments)
    # hvo_seq.major_and_minor_grid_lines
    hvo_seq.to_html_plot(show_figure=True)