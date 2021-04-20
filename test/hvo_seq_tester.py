from hvo_sequence.hvo_seq import HVO_Sequence
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING

import numpy as np


if __name__ == "__main__":

    # Create an instance of a HVO_Sequence
    hvo_seq = HVO_Sequence(drum_mapping=ROLAND_REDUCED_MAPPING)

    # Add two time_signatures
    hvo_seq.add_time_signature(0, 4, 4, [4])
    hvo_seq.add_time_signature(13, 6, 8, [3,2])

    # Add two tempos
    hvo_seq.add_tempo(0, 50)

    hvo_seq.add_tempo(12, 20)  # Tempo Change at the beginning of second bar

    # Create a random hvo
    hits = np.random.randint(0, 2, (36, 9))
    vels = hits * np.random.rand(36, 9)
    offs = hits * (np.random.rand(36, 9) - 0.5)

    # Add hvo score to hvo_seq instance
    hvo_seq.hvo = np.concatenate((hits, vels, offs), axis=1)

    # Plot, synthesize and export to midi
    #hvo_seq.to_html_plot(show_figure=True)
    #hvo_seq.save_audio()
    #hvo_seq.save_hvo_to_midi()

    # print(hvo_seq.steps_per_beat_per_segments)
    # print(hvo_seq.grid_lines)
    # print(hvo_seq.grid_lines.shape)
    # print(hvo_seq.n_beats_per_segments)
    # hvo_seq.major_and_minor_grid_lines
    #

    #STFT
    print(hvo_seq.stft())
    hvo_seq.save_spectrogram()

