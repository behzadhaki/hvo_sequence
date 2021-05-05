import numpy as np
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import find_peaks
import math
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING, Groove_Toolbox_5Part_keymap, Groove_Toolbox_3Part_keymap


class CompleteFeatureExtractor:
    """
    Calculates all of the following features from an hvo_seq object

    # GrooveToolbox.RhythmFeatures
        # --------------------------------------
        total_autocorrelation_curve
        combined_syncopation
        polyphonic_syncopation
        low_syncopation
        mid_syncopation
        high_syncopation
        low_density
        mid_density
        high_density
        total_density
        hiness
        midness
        lowness
        hisyncness
        midsyncness
        lowsyncness
        autocorrelation_skew
        autocorrelation_max_amplitude
        autocorrelation_centroid
        autocorrelation_harmonicity
        total_symmetry
        total_average_intensity
        total_weak_to_strong_ratio
        total_complexity

    # GrooveToolbox.MicrotimingFeatures
        # --------------------------------------
        swingness
        is_swung
        laidbackness
        timing_accuracy
    """

    def __init__(self, hvo_seq, _5kitparts_map, _3kitparts_map):
        self.features_dictionary = None

        # #############################################################
        #                   GrooveToolBox Features
        #        Create an Instance of the GrooveToolbox to calculate
        #        the rhythm and microtiming info implemented in the toolbox
        # #############################################################

        # Create an instance of GrooveToolbox
        self.groove_toolbox = GrooveToolbox(
            hvo_seq, _5kitparts_map=_5kitparts_map,
            _3kitparts_map=_3kitparts_map)

        # #############################################################
        #                   Spectral Features
        #        Create and instance of the SpectralFeatures to calculate
        #        the spectral features
        # #############################################################

        # Create an instance of the SpectralFeatures class
        self.spectral_features = SpectralFeatures(hvo_seq)

    def get_feature_dictionary(self, feature_list=[None]):
        # If feature_list is [None] calculate and compile all features
        # otherwise only do so for the specified features in feature_list
        self.groove_toolbox.RhythmFeatures.calculate_all_features(feature_list=feature_list)
        self.groove_toolbox.MicrotimingFeatures.calculate_all_features(feature_list=feature_list)

        self.features_dictionary = {
            # GrooveToolbox.RhythmFeatures
            # --------------------------------------
            "combined_syncopation": self.groove_toolbox.RhythmFeatures.combined_syncopation,
            "polyphonic_syncopation": self.groove_toolbox.RhythmFeatures.polyphonic_syncopation,
            "low_syncopation": self.groove_toolbox.RhythmFeatures.low_syncopation,
            "mid_syncopation": self.groove_toolbox.RhythmFeatures.mid_syncopation,
            "high_syncopation": self.groove_toolbox.RhythmFeatures.high_syncopation,
            "low_density": self.groove_toolbox.RhythmFeatures.low_density,
            "mid_density": self.groove_toolbox.RhythmFeatures.mid_density,
            "high_density": self.groove_toolbox.RhythmFeatures.high_density,
            "total_density": self.groove_toolbox.RhythmFeatures.total_density,
            "hiness": self.groove_toolbox.RhythmFeatures.hiness,
            "midness": self.groove_toolbox.RhythmFeatures.midness,
            "lowness": self.groove_toolbox.RhythmFeatures.lowness,
            "hisyncness": self.groove_toolbox.RhythmFeatures.hisyncness,
            "midsyncness": self.groove_toolbox.RhythmFeatures.midsyncness,
            "lowsyncness": self.groove_toolbox.RhythmFeatures.lowsyncness,
            "total_autocorrelation_curve": self.groove_toolbox.RhythmFeatures.total_autocorrelation_curve,
            "autocorrelation_skew": self.groove_toolbox.RhythmFeatures.autocorrelation_skew,
            "autocorrelation_max_amplitude": self.groove_toolbox.RhythmFeatures.autocorrelation_max_amplitude,
            "autocorrelation_centroid": self.groove_toolbox.RhythmFeatures.autocorrelation_centroid,
            "autocorrelation_harmonicity": self.groove_toolbox.RhythmFeatures.autocorrelation_harmonicity,
            "total_symmetry": self.groove_toolbox.RhythmFeatures.total_symmetry,
            "total_average_intensity": self.groove_toolbox.RhythmFeatures.total_average_intensity,
            "total_weak_to_strong_ratio": self.groove_toolbox.RhythmFeatures.total_weak_to_strong_ratio,
            "total_complexity": self.groove_toolbox.RhythmFeatures.total_complexity,

            # GrooveToolbox.MicrotimingFeatures
            # --------------------------------------
            "swingness": self.groove_toolbox.MicrotimingFeatures.swingness,
            "is_swung": self.groove_toolbox.MicrotimingFeatures.is_swung,
            "laidbackness": self.groove_toolbox.MicrotimingFeatures.laidbackness,
            "timing_accuracy": self.groove_toolbox.MicrotimingFeatures.timing_accuracy

            # Spectral Features
            # --------------------------------------
            # "onset_confidence": None
            # todo: add features here when the method is implemented
        }

        compiled_features = {}
        if feature_list != [None]:
            for key in feature_list:
                compiled_features[key]=self.features_dictionary[key]
        else:
            compiled_features = self.features_dictionary
        return compiled_features

#######################################################################
#                  Audio/Spectral Based Features
#
#        to implement
#               1. Multi-voice onset confidence
#               2. Tapped onset confidence
#               3. ConstQ Transformed Spectral Flux
#
#######################################################################


# todo: Implement the audio based features here
class SpectralFeatures:
    def __init__(self, hvo_seq):
        self.spectral_features_dict = None
        return

    def calculate_all_features(self):
        self.spectral_features_dict = dict()
        return self.spectral_features_dict


#######################################################################
#        Rhythmic and Micro-timing Features from GrooveToolbox
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

def get_tgt_map_index_for_src_map(src_map, tgt_map):
    """
    Finds the corresponding voice group index for
    :param src_map:   a drum_mapping dictionary
    :param tgt_map:     a drum_mapping dictionary
    :return: list of same len as src_map. Each element in returned list is the corresponding voice group to be used in
             tgt_map for each of the voice groups in the src_map

            Example:
            if src_map = ROLAND_REDUCED_MAPPING and tgt_map = Groove_Toolbox_5Part_keymap
            the return will be [0, 1, 2, 3, 4, 4, 4, 3, 2]
            this means that kicks are to be mapped to kick
                            snares are to be mapped to snares
                            c_hat and rides are to be mapped to the same group (closed)
                            o_hat and crash are to be mapped to the same group (open)
                            low. mid. hi Toms are to be mapped to the same group (toms)

    """
    src_ix_to_tgt_ix_map = []
    for src_voice_ix, src_voice_midi_list in enumerate(src_map.values()):
        corresponding_tgt_indices = []
        for tgt_voice_ix, tgt_voice_midi_list in enumerate(tgt_map.values()):
            if src_voice_midi_list[0] in tgt_voice_midi_list:
                corresponding_tgt_indices.append(tgt_voice_ix)
        src_ix_to_tgt_ix_map.append(max(corresponding_tgt_indices, key=corresponding_tgt_indices.count))

    return src_ix_to_tgt_ix_map


def convert_groove(groove, src_map, tgt_map):
    # Groove is a velocity matrix here
    src_ix_to_tgt_ix_map = get_tgt_map_index_for_src_map(src_map, tgt_map)

    n_steps = groove.shape[0]
    target_groove = np.zeros((n_steps, len(tgt_map.keys())))

    for src_voice_ix, groove_row in enumerate(np.transpose(groove)):
        tgt_voice_ix = src_ix_to_tgt_ix_map[src_voice_ix]
        target_groove[:, tgt_voice_ix] = np.clip(target_groove[:, tgt_voice_ix] + groove_row, 0, 1)

    return target_groove

def convert_groove_toolbox(grooveToolbox, src_map, tgt_map):
    # Converts the score in the groovetoolbox instance
    src_ix_to_tgt_ix_map = get_tgt_map_index_for_src_map(src_map, tgt_map)

    n_steps = grooveToolbox.groove_all_parts.shape[0]
    target_groove = np.zeros((n_steps, len(tgt_map.keys())))

    for src_voice_ix, groove_row in enumerate(np.transpose(grooveToolbox.groove_all_parts)):
        tgt_voice_ix = src_ix_to_tgt_ix_map[src_voice_ix]
        target_groove[:, tgt_voice_ix] = np.clip(target_groove[:, tgt_voice_ix] + groove_row, 0, 1)

    grooveToolbox.groove_all_parts = target_groove
    return grooveToolbox

class GrooveToolbox:
    def __init__(self, hvo_seq, _5kitparts_map, _3kitparts_map, extract_features=False,
                 velocity_type="Regular", name="Groove"):
        # extractFeatures - if True (default), extract all features upon groove creation.
        # Set false if you don't need all features - instead retrieve as and when you need them

        assert len(hvo_seq.time_signatures) == 1, \
            "Currently time signature changes are not supported for feature extractions"

        assert hvo_seq.time_signatures[0].numerator == 4 and hvo_seq.time_signatures[0].denominator == 4, \
            "Currently only 4/4 time_signatures are supported for feature extractions"

        assert hvo_seq.time_signatures[0].numerator == 4 and hvo_seq.time_signatures[0].denominator == 4, \
            "Currently only 16th note quantization (beat_division_factor=[4]) is supported  for feature extractions"

        assert hvo_seq.hvo.shape[0] == 32 or hvo_seq.hvo.shape[0] == 16, \
            "Currently only 16 or 32 time steps are supported"

        self.groove_all_parts = hvo_seq.get("v")
        self.timing_matrix = hvo_seq.get("o", offsets_in_ms=True)

        if hvo_seq.hvo.shape[0] == 16:
            self.groove_all_parts = np.concatenate([self.groove_all_parts, self.groove_all_parts])
            self.timing_matrix = np.concatenate([self.timing_matrix, self.timing_matrix])

        tempo = hvo_seq.tempos[0].qpm

        self.all_parts_map = hvo_seq.drum_mapping
        self._5kitParts_map = _5kitparts_map
        self._3kitParts_map = _3kitparts_map

        self.name = name

        if velocity_type == "None":
            self.groove_all_parts = np.ceil(self.groove_all_parts)
        if velocity_type == "Regular":
            pass
        if velocity_type == "Transform":
            self.groove_all_parts = np.power(self.groove_all_parts, 0.2)

        self.groove_5_parts = convert_groove(self.groove_all_parts, self.all_parts_map, self._5kitParts_map)
        self.groove_3_parts = convert_groove(self.groove_all_parts, self.all_parts_map, self._3kitParts_map)

        self.RhythmFeatures = RhythmFeatures(self.groove_all_parts, self.groove_5_parts, self.groove_3_parts,
                                             self.all_parts_map, self._5kitParts_map, self._3kitParts_map)
        self.MicrotimingFeatures = MicrotimingFeatures(self.timing_matrix, tempo)

        if extract_features:
            self.RhythmFeatures.calculate_all_features()
            self.MicrotimingFeatures.calculate_all_features()

    def reduce_groove(self):
        # Remove ornamentation from a groove to return a simplified representation of the rhythm structure
        # change salience profile for different metres etc
        metrical_profile = [0, -2, -1, -2, 0, -2, -1, -2, -0, -2, -1, -2, -0, -2, -1, -2,
                            0, -2, -1, -2, 0, -2, -1, -2, 0, -2, -1, -2, 0, -2, -1, -2]
        self.reduced_groove = np.zeros(self.groove_all_parts.shape)
        for i in range(len(self.all_parts_map.keys())):     # number of parts to reduce
            self.reduced_groove[:, i] = self._reduce_part(self.groove_all_parts[:, i], metrical_profile)
        rows_to_remove = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 30, 31]
        self.reduced_groove = np.delete(self.reduced_groove, rows_to_remove, axis=0)
        return self.reduced_groove

    def _reduce_part(self, part, metrical_profile):
        length = self.groove_all_parts.shape[0]
        for i in range(length):
            if part[i] <= 0.4:
                part[i] = 0
        for i in range(length):
            if part[i] != 0.:  # hit detected - must be figural or density transform - on pulse i.
                for k in range(-3, i):  # iterate through all previous events up to i.
                    if part[k] != 0. and metrical_profile[k] < metrical_profile[i]:
                        # if there is a preceding event in a weaker pulse k (this is to be removed)

                        # groove[k,0] then becomes k, and can either be density of figural
                        previous_event_index = 0
                        for l in range(0, k):  # find strongest level pulse before k, with no events between m and k
                            if part[l] != 0.:  # find an event if there is one
                                previous_event_index = l
                            else:
                                previous_event_index = 0
                        m = max(metrical_profile[
                                previous_event_index:k])  # find the strongest level between previous event index and k.
                        # search for largest value in salience profile list between range l+1 and k-1. this is m.
                        if m <= k:  # density if m not stronger than k
                            part[k] = 0  # to remove a density transform just remove the note
                        if m > k:  # figural transform
                            part[m] = part[k]  # need to shift note forward - k to m.
                            part[k] = 0  # need to shift note forward - k to m.
            if part[i] == 0:
                for k in range(-3, i):
                    if part[k] != 0. and metrical_profile[k] < metrical_profile[i]:  # syncopation detected
                        part[i] = part[k]
                        part[k] = 0.0
        return part


class RhythmFeatures:
    def __init__(self, groove_all_parts, groove_5_parts, groove_3_parts,
                 all_parts_map, _5kitparts_map, _3kitparts_map):

        self.all_parts_map = all_parts_map
        self._5kitParts_map = _5kitparts_map
        self._3kitParts_map = _3kitparts_map

        self.groove_all_parts = groove_all_parts
        self.groove_5_parts = groove_5_parts
        self.groove_3_parts = groove_3_parts

        self.total_autocorrelation_curve = None
        self.combined_syncopation = None
        self.polyphonic_syncopation = None
        self.low_syncopation = None
        self.mid_syncopation = None
        self.high_syncopation = None
        self.low_density = None
        self.mid_density = None
        self.high_density = None
        self.total_density = None
        self.hiness = None
        self.midness = None
        self.lowness = None
        self.hisyncness = None
        self.midsyncness = None
        self.lowsyncness = None
        self.autocorrelation_skew = None
        self.autocorrelation_max_amplitude = None
        self.autocorrelation_centroid = None
        self.autocorrelation_harmonicity = None
        self.total_symmetry = None
        self.total_average_intensity = None
        self.total_weak_to_strong_ratio = None
        self.total_complexity = None

        # todo: Do I want to list names of class variables in here? So user can see them easily?

    def calculate_all_features(self, feature_list=[None]):
        # Gets requested features or all (if feature_list==[None])
        #
        self.combined_syncopation = self.get_combined_syncopation() if "combined_syncopation" in feature_list or feature_list==[None] else None
        self.polyphonic_syncopation = self.get_polyphonic_syncopation() if "polyphonic_syncopation" in feature_list or feature_list == [None] else None
        self.low_syncopation = self.get_low_syncopation() if "low_syncopation" in feature_list or feature_list == [None] else None
        self.mid_syncopation = self.get_mid_syncopation() if "mid_syncopation" in feature_list or feature_list == [None] else None
        self.high_syncopation = self.get_high_syncopation() if "high_syncopation" in feature_list or feature_list == [None] else None
        self.low_density = self.get_low_density() if "low_density" in feature_list or feature_list == [None] else None
        self.mid_density = self.get_mid_density() if "mid_density" in feature_list or feature_list == [None] else None
        self.high_density = self.get_high_density() if "high_density" in feature_list or feature_list == [None] else None
        self.total_density = self.get_total_density() if "total_density" in feature_list or feature_list == [None] else None
        self.hiness = self.get_hiness() if "hiness" in feature_list or feature_list == [None] else None
        self.midness = self.get_midness() if "midness" in feature_list or feature_list == [None] else None
        self.lowness = self.get_lowness() if "lowness" in feature_list or feature_list == [None] else None
        self.hisyncness = self.get_hisyncness() if "hisyncness" in feature_list or feature_list == [None] else None
        self.midsyncness = self.get_midsyncness() if "midsyncness" in feature_list or feature_list == [None] else None
        self.lowsyncness = self.get_lowsyncness() if "lowsyncness" in feature_list or feature_list == [None] else None
        self.autocorrelation_skew = self.get_autocorrelation_skew() if "autocorrelation_skew" in feature_list or feature_list == [None] else None
        self.autocorrelation_max_amplitude = self.get_autocorrelation_max_amplitude() if "autocorrelation_max_amplitude" in feature_list or feature_list == [None] else None
        self.autocorrelation_centroid = self.get_autocorrelation_centroid() if "autocorrelation_centroid" in feature_list or feature_list == [None] else None
        self.autocorrelation_harmonicity = self.get_autocorrelation_harmonicity() if "autocorrelation_harmonicity" in feature_list or feature_list == [None] else None
        self.total_symmetry = self.get_total_symmetry() if "total_symmetry" in feature_list or feature_list == [None] else None
        self.total_average_intensity = self.get_total_average_intensity() if "total_average_intensity" in feature_list or feature_list == [None] else None
        self.total_weak_to_strong_ratio = self.get_total_weak_to_strong_ratio() if "total_weak_to_strong_ratio" in feature_list or feature_list == [None] else None
        self.total_complexity = self.get_total_complexity() if "total_complexity" in feature_list or feature_list == [None] else None

    def get_all_features(self):
        return np.hstack([self.combined_syncopation,self.polyphonic_syncopation,self.low_syncopation, self.mid_syncopation,
                          self.high_syncopation, self.total_weak_to_strong_ratio, self.low_density,  self.mid_density, self.high_density, self.total_density,
                          self.hiness, self.hisyncness, self.total_average_intensity, self.total_complexity, self.autocorrelation_skew, self.autocorrelation_max_amplitude,
                          self.autocorrelation_centroid, self.autocorrelation_harmonicity, self.total_symmetry])

    def print_all_features(self):
        print("\n   Rhythm features:")
        print("Combined Mono Syncopation = " + str(self.combined_syncopation))
        print("Polyphonic Syncopation = " + str(self.polyphonic_syncopation))
        print("Weak to Strong Ratio = " + str(self.total_weak_to_strong_ratio))
        print("Low Syncopation = " + str(self.low_syncopation))
        print("Mid Syncopation = " + str(self.mid_syncopation))
        print("High Syncopation = " + str(self.high_syncopation))
        print("Total Density = " + str(self.total_density))
        print("hiness = " + str(self.hiness))
        print("midness = " + str(self.midness))
        print("lowness = " + str(self.lowness))
        print("hisyncness = " + str(self.hisyncness))
        print("midsyncness = " + str(self.midsyncness))
        print("lowsyncness = " + str(self.lowsyncness))
        print("Low Density = " + str(self.low_density))
        print("Mid Density = " + str(self.mid_density))
        print("High Density = " + str(self.high_density))
        print("Total Complexity = " + str(self.total_complexity))
        print("Total Average Intensity = " + str(self.total_average_intensity))
        print("Autocorrelation Skewness = " + str(self.autocorrelation_skew))
        print("Autocorrelation Max Amplitude = " + str(self.autocorrelation_max_amplitude))
        print("Autocorrelation Centroid = " + str(self.autocorrelation_centroid))
        print("Autocorrelation Harmonicity = " + str(self.autocorrelation_harmonicity))
        print("Symmetry = " + str(self.total_symmetry))

    def get_combined_syncopation(self):
        # Calculate syncopation as summed across all kit parts.
        # Tested - working correctly (12/3/20)

        self.combined_syncopation = 0.0
        for i in range(self.groove_all_parts.shape[1]):
            self.combined_syncopation += self.get_syncopation_1part(self.groove_all_parts[:,i])
        return self.combined_syncopation

    def get_polyphonic_syncopation(self):
        # Calculate syncopation using Witek syncopation distance - modelling syncopation between instruments
        # Works on semiquaver and quaver levels of syncopation
        # todo: Normalize...?

        metrical_profile = [0, -3, -2, -3, -1, -3, -2, -3, -1, -3, -2, -3, -1, -3, -2, -3,
                           0, -3, -2, -3, -1, -3, -2, -3, -1, -3, -2, -3, -1, -3, -2, -3]

        low = self.groove_3_parts[:,0]
        mid = self.groove_3_parts[:,1]
        high = self.groove_3_parts[:,2]

        total_syncopation = 0

        for i in range(len(low)):
            kick_syncopation = self._get_kick_syncopation(low, mid, high, i, metrical_profile)
            snare_syncopation = self._get_snare_syncopation(low, mid, high, i, metrical_profile)
            total_syncopation += kick_syncopation * low[i]
            total_syncopation += snare_syncopation * mid[i]
        return total_syncopation

    def _get_kick_syncopation(self, low, mid, high, i, metrical_profile):
        # Find instances  when kick syncopates against hi hat/snare on the beat.
        # For use in polyphonic syncopation feature

        kick_syncopation = 0
        k = 0
        next_hit = ""
        if low[i] == 1 and low[(i + 1) % 32] !=1 and low[(i+2) % 32] != 1:
            for j in i + 1, i + 2, i + 3, i + 4: #look one and two steps ahead only - account for semiquaver and quaver sync
                if mid[(j % 32)] == 1 and high[(j % 32)] != 1:
                    next_hit = "Mid"
                    k = j % 32
                    break
                elif high[(j % 32)] == 1 and mid[(j % 32)] != 1:
                    next_hit = "High"
                    k = j % 32
                    break
                elif high[(j % 32)] == 1 and mid[(j % 32)] == 1:
                    next_hit = "MidAndHigh"
                    k = j % 32
                    break
                # if both next two are 0 - next hit == rest. get level of the higher level rest
            if mid[(i+1)%32] + mid[(i+2)%32] == 0.0 and high[(i+1)%32] + [(i+2)%32] == 0.0:
                next_hit = "None"

            if next_hit == "MidAndHigh":
                if metrical_profile[k] >= metrical_profile[i]:  # if hi hat is on a stronger beat - syncopation
                    difference = metrical_profile[k] - metrical_profile[i]
                    kick_syncopation = difference + 2
            elif next_hit == "Mid":
                if metrical_profile[k] >= metrical_profile[i]:  # if hi hat is on a stronger beat - syncopation
                    difference = metrical_profile[k] - metrical_profile[i]
                    kick_syncopation = difference + 2
            elif next_hit == "High":
                if metrical_profile[k] >= metrical_profile[i]:
                    difference = metrical_profile[k] - metrical_profile[i]
                    kick_syncopation = difference + 5
            elif next_hit == "None":
                if metrical_profile[k] > metrical_profile[i]:
                    difference = max(metrical_profile[(i + 1) % 32], metrical_profile[(i + 2) % 32]) - metrical_profile[i]
                    kick_syncopation = difference + 6 # if rest on a stronger beat - one stream sync, high sync value
        return kick_syncopation

    def _get_snare_syncopation(self, low, mid, high, i, metrical_profile):
        # Find instances  when snare syncopates against hi hat/kick on the beat
        # For use in polyphonic syncopation feature

        snare_syncopation = 0
        next_hit = ""
        k = 0
        if mid[i] == 1 and mid[(i + 1) % 32] !=1 and mid[(i+2) % 32] != 1:
            for j in i + 1, i + 2, i + 3, i + 4: #look one and 2 steps ahead only
                if low[(j % 32)] == 1 and high[(j % 32)] != 1:
                    next_hit = "Low"
                    k = j % 32
                    break
                elif high[(j % 32)] == 1 and low[(j % 32)] != 1:
                    next_hit = "High"
                    k = j % 32
                    break
                elif high[(j % 32)] == 1 and low[(j % 32)] == 1:
                    next_hit = "LowAndHigh"
                    k = j % 32
                    break
            if low[(i+1)%32] + low[(i+2)%32] == 0.0 and high[(i+1)%32] + [(i+2)%32] == 0.0:
                next_hit = "None"

            if next_hit == "LowAndHigh":
                if metrical_profile[k] >= metrical_profile[i]:
                    difference = metrical_profile[k] - metrical_profile[i]
                    snare_syncopation = difference + 1  # may need to make this back to 1?)
            elif next_hit == "Low":
                if metrical_profile[k] >= metrical_profile[i]:
                    difference = metrical_profile[k] - metrical_profile[i]
                    snare_syncopation = difference + 1
            elif next_hit == "High":
                if metrical_profile[k] >= metrical_profile[i]:  # if hi hat is on a stronger beat - syncopation
                    difference = metrical_profile[k] - metrical_profile[i]
                    snare_syncopation = difference + 5
            elif next_hit == "None":
                if metrical_profile[k] > metrical_profile[i]:
                    difference = max(metrical_profile[(i + 1) % 32], metrical_profile[(i + 2) % 32]) - metrical_profile[i]
                    snare_syncopation = difference + 6 # if rest on a stronger beat - one stream sync, high sync value
        return snare_syncopation

    def get_syncopation_1part(self, part):
        # Using Longuet-Higgins  and  Lee 1984 metric profile, get syncopation of 1 monophonic line.
        # Assumes it's a drum loop - loops round.
        # Normalized against maximum syncopation: syncopation score of pattern with all pulses of lowest metrical level
        # at maximum amplitude (=30 for 2 bar 4/4 loop)

        metrical_profile = [5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1,
                                   5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1]
        max_syncopation = 30.0
        syncopation = 0.0
        for i in range(len(part)):
            if part[i] != 0:
                if part[(i + 1) % 32] == 0.0 and metrical_profile[(i + 1) % 32] > metrical_profile[i]:
                    syncopation = float(syncopation + (
                    abs(metrical_profile[(i + 1) % 32] - metrical_profile[i]))) # * part[i])) #todo: velocity here?

                elif part[(i + 2) % 32] == 0.0 and metrical_profile[(i + 2) % 32] > metrical_profile[i]:
                    syncopation = float(syncopation + (
                    abs(metrical_profile[(i + 2) % 32] - metrical_profile[i]))) # * part[i]))
        return syncopation / max_syncopation

    def get_average_intensity(self, part):
        # Get average loudness for any single part or group of parts. Will return 1 for binary loop, otherwise calculate
        # based on velocity mode chosen (transform or regular)

        # first get all non-zero hits. then divide by number of hits

        hit_indexes = np.nonzero(part)
        total = 0.0
        hit_count = np.count_nonzero(part)

        for i in range(hit_count):
            if len(hit_indexes) > 1:
                index = hit_indexes[0][i], hit_indexes[1][i]
            else:
                index = hit_indexes[0][i]
            total += part[index]
        average = total / hit_count
        return average

    def get_total_average_intensity(self):
        # Get average loudness for every hit in a loop
        self.total_average_intensity = self.get_average_intensity(self.groove_all_parts)
        return self.total_average_intensity

    def get_weak_to_strong_ratio(self, part):
        weak_hit_count = 0.0
        strong_hit_count = 0.0

        strong_positions = [0,4,8,12,16,20,24,28]
        weak_positions = [1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,21,22,23,25,26,27,29,30,31]

        hits_count = np.count_nonzero(part)
        hit_indexes = np.nonzero(part)
        for i in range(hits_count):
            if len(hit_indexes) > 1:
                index = hit_indexes[0][i], hit_indexes[1][i]
            else:
                index = [hit_indexes[0][i]]
            if index[0] in strong_positions:
                strong_hit_count += part[index]
            if index[0] in weak_positions:
                weak_hit_count += part[index]
        weakToStrongRatio = weak_hit_count/strong_hit_count
        return weakToStrongRatio

    def get_total_weak_to_strong_ratio(self): #todo: test
        self.total_weak_to_strong_ratio = self.get_weak_to_strong_ratio(self.groove_all_parts)
        return self.total_weak_to_strong_ratio

    def get_low_syncopation(self):
        # Get syncopation of low part (kick drum)
        self.low_syncopation = self.get_syncopation_1part(self.groove_3_parts[:, 0])
        return self.low_syncopation

    def get_mid_syncopation(self):
        # Get syncopation of mid parts - summed snare and tom parts
        midParts = self.groove_3_parts[:, 1]
        self.mid_syncopation = self.get_syncopation_1part(midParts)
        return self.mid_syncopation

    def get_high_syncopation(self):
        # Get syncopation of high parts - summed cymbals
        highParts = self.groove_3_parts[:, 2]
        self.high_syncopation = self.get_syncopation_1part(highParts)
        return self.high_syncopation

    def get_density(self, part): #todo: rename to be consistent with other features? that use 1 part vs all parts functions
        # Get density of any single kit part or part group. Difference to total density is that you divide by
        # number of metrical steps, instead of total number of possible onsets in the pattern

        step_count = part.shape[0]
        onset_count = np.count_nonzero(np.ceil(part) == 1)
        average_velocity = part[part!=0.0].mean()

        if np.isnan(average_velocity):
            average_velocity = 0.0
        density = average_velocity * float(onset_count) / float(step_count)
        return density

    def get_low_density(self):
        # Get density of low part (kick)
        src_ix_to_tgt_ix_map = np.array(get_tgt_map_index_for_src_map(self.all_parts_map, self._3kitParts_map))
        low_indices = np.where(src_ix_to_tgt_ix_map == 0)
        low_parts = list()
        for low_index in low_indices:
            low_parts.append(self.groove_all_parts[:, low_index])
        low_parts = np.vstack([low_parts])
        self.low_density = self.get_density(low_parts.T)

        return self.low_density

    def get_mid_density(self):
        # Get density of mid parts (toms and snare)
        src_ix_to_tgt_ix_map = np.array(get_tgt_map_index_for_src_map(self.all_parts_map, self._3kitParts_map))
        mid_indices = np.where(src_ix_to_tgt_ix_map == 1)
        mid_parts = list()
        for mid_index in mid_indices:
            mid_parts.append(self.groove_all_parts[:, mid_index])
        mid_parts = np.vstack(mid_parts)
        self.mid_density = self.get_density(mid_parts.T)
        return self.mid_density

    def get_high_density(self):
        # Get density of high parts (cymbals)
        src_ix_to_tgt_ix_map = np.array(get_tgt_map_index_for_src_map(self.all_parts_map, self._3kitParts_map))
        hi_indices = np.where(src_ix_to_tgt_ix_map == 2)
        hi_parts = list()
        for hi_index in hi_indices:
            hi_parts.append(self.groove_all_parts[:, hi_index])
        hi_parts = np.vstack(hi_parts)
        self.high_density = self.get_density(hi_parts.T)
        return self.high_density

    def get_total_density(self):
        # Get total density calculated over 10 parts.
        # Total density = number of onsets / number of possible onsets (= length of pattern x 10)
        # Return values tend to be very low for this, due to high numbers of parts meaning sparse
        # matrices.

        total_step_count = self.groove_all_parts.size
        onset_count = np.count_nonzero(np.ceil(self.groove_all_parts) == 1)
        average_velocity = self.groove_all_parts[self.groove_all_parts!=0.0].mean()
        self.total_density = average_velocity * float(onset_count) / float(total_step_count)
        return self.total_density

    def get_lowness(self):
        # Extended by BH
        # might need to x total density by 10 as it's not summed over one line
        self.lowness = (float(self.get_low_density()) / float(self.get_total_density()))
        return self.lowness

    def get_midness(self):
        # Extended by BH
        # might need to x total density by 10 as it's not summed over one line
        self.midness = (float(self.get_mid_density()) / float(self.get_total_density()))
        return self.midness

    def get_hiness(self):
        # might need to x total density by 10 as it's not summed over one line
        self.hiness = (float(self.get_high_density()) / float(self.get_total_density()))
        return self.hiness

    def get_lowsyncness(self):
        low_density = self.get_low_density()

        if low_density != 0.:
            self.lowsyncness = float(self.get_low_syncopation()) / float(low_density)
        else:
            self.lowsyncness = 0
        return self.lowsyncness

    def get_midsyncness(self):
        mid_density = self.get_mid_density()

        if mid_density != 0.:
            self.midsyncness = float(self.get_mid_syncopation()) / float(mid_density)
        else:
            self.midsyncness = 0
        return self.midsyncness

    def get_hisyncness(self):
        high_density = self.get_high_density()

        if high_density != 0.:
            self.hisyncness = float(self.get_high_syncopation()) / float(high_density)
        else:
            self.hisyncness = 0
        return self.hisyncness

    def get_complexity_1part(self, part):
        # Get complexity of one part. Calculated following Sioros and Guedes (2011) as
        # combination of density and syncopation
        # Uses monophonic syncopation measure
        density = self.get_density(part)
        syncopation = self.get_syncopation_1part(part)
        complexity = math.sqrt(pow(density, 2) + pow(syncopation, 2))
        return complexity

    def get_total_complexity(self):
        density = self.get_total_density()
        syncopation = self.get_combined_syncopation()
        self.total_complexity = math.sqrt(pow(density, 2) + pow(syncopation, 2))
        return self.total_complexity

    def _get_autocorrelation_curve(self, part):
        # todo:replace this autocorrelation function for a better one
        # Return autocorrelation curve for a single part.
        # Uses autocorrelation plot function within pandas

        plt.figure()
        ax = autocorrelation_plot(part)
        autocorrelation = ax.lines[5].get_data()[1]
        plt.plot(range(1, self.groove_all_parts.shape[0]+1),
                 autocorrelation)  # plots from 1 to 32 inclusive - autocorrelation starts from 1 not 0 - 1-32
        plt.clf()
        plt.cla()
        plt.close()
        return np.nan_to_num(autocorrelation)

    def get_total_autocorrelation_curve(self):
        # Get autocorrelation curve for all parts summed.

        self.total_autocorrelation_curve = 0.0
        for i in range(self.groove_all_parts.shape[1]):
            self.total_autocorrelation_curve += self._get_autocorrelation_curve(self.groove_all_parts[:, i])

        # ax = autocorrelation_plot(self.total_autocorrelation_curve)
        # plt.figure()
        # plt.plot(range(1,33),self.total_autocorrelation_curve)
        return self.total_autocorrelation_curve

    def get_autocorrelation_skew(self):
        self.total_autocorrelation_curve = self.get_total_autocorrelation_curve()
        # Get skewness of autocorrelation curve
        if self.total_autocorrelation_curve is not None:
            pass
        else:
            self.total_autocorrelation_curve = self.get_total_autocorrelation_curve()

        self.autocorrelation_skew = stats.skew(self.total_autocorrelation_curve)
        return self.autocorrelation_skew

    def get_autocorrelation_max_amplitude(self):
        # Get maximum amplitude of autocorrelation curve
        self.total_autocorrelation_curve = self.get_total_autocorrelation_curve()

        self.autocorrelation_max_amplitude = self.total_autocorrelation_curve.max()
        return self.autocorrelation_max_amplitude

    def get_autocorrelation_centroid(self):
        # Like spectral centroid - weighted mean of frequencies in the signal, magnitude = weights.
        centroid_sum = 0
        total_weights = 0
        self.total_autocorrelation_curve = self.get_total_autocorrelation_curve()

        for i in range(self.total_autocorrelation_curve.shape[0]):
            # half wave rectify
            addition = self.total_autocorrelation_curve[i] * i   # sum all periodicity's in the signal
            if addition >= 0:
                total_weights += self.total_autocorrelation_curve[i]
                centroid_sum += addition
        if total_weights != 0:
            self.autocorrelation_centroid = centroid_sum / total_weights
        else:
            self.autocorrelation_centroid = self.groove_all_parts.shape[0] / 2
        return self.autocorrelation_centroid

    def get_autocorrelation_harmonicity(self):
        # Autocorrelation Harmonicity adapted from Lartillot et al. 2008
        self.total_autocorrelation_curve = self.get_total_autocorrelation_curve()

        alpha = 0.15
        rectified_autocorrelation = self.total_autocorrelation_curve
        for i in range(self.total_autocorrelation_curve.shape[0]):
            if self.total_autocorrelation_curve[i] < 0:
                rectified_autocorrelation[i] = 0

        # weird syntax due to 2.x/3.x compatibility issues here todo: rewrite for 3.x
        peaks = np.asarray(find_peaks(rectified_autocorrelation))
        peaks = peaks[0] + 1  # peaks = lags

        inharmonic_sum = 0.0
        inharmonic_peaks = []
        for i in range(len(peaks)):
            remainder1 = 16 % peaks[i]
            if remainder1 > 16 * alpha and remainder1 < 16 * (1-alpha):
                inharmonic_sum += rectified_autocorrelation[peaks[i] - 1]  # add magnitude of inharmonic peaks
                inharmonic_peaks.append(rectified_autocorrelation[i])

        harmonicity = math.exp((-0.25 * len(peaks) * inharmonic_sum / float(rectified_autocorrelation.max())))
        return harmonicity

    def _get_symmetry(self, part):
        # Calculate symmetry for any number of parts.
        # Defined as the the number of onsets that appear in the same positions in the first and second halves
        # of the pattern, divided by the total number of onsets in the pattern. As perfectly symmetrical pattern
        # would have a symmetry of 1.0

        symmetry_count = 0.0
        part1, part2 = np.split(part, 2)
        for i in range(part1.shape[0]):
            for j in range(part1.shape[1]):
                if part1[i, j] != 0.0 and part2[i, j] != 0.0:
                    symmetry_count += (1.0 - abs(part1[i, j] - part2[i, j]))
        symmetry = symmetry_count*2.0 / np.count_nonzero(part)
        return symmetry

    def get_total_symmetry(self):
        # Get total symmetry of pattern. Defined as the number of onsets that appear in the same positions in the first
        # and second halves of the pattern, divided by total number of onsets in the pattern.

        self.total_symmetry = self._get_symmetry(self.groove_all_parts)
        return self.total_symmetry


class MicrotimingFeatures:
    def __init__(self, microtiming_matrix, tempo):
        self.microtiming_matrix = microtiming_matrix

        self.tempo = tempo
        self.average_timing_matrix = self.get_average_timing_deviation()
        self._get_swing_info()
        self.microtiming_event_profile = np.hstack(
            [self._getmicrotiming_event_profile_1bar(self.microtiming_matrix[0:16]),
             self._getmicrotiming_event_profile_1bar(self.microtiming_matrix[16:])])
        self.laidback_events = self._get_laidback_events()
        self.pushed_events = self._get_pushed_events()
        self.is_swung = self.check_if_swung()

        self.laidbackness = None
        self.timing_accuracy = None

    def calculate_all_features(self, feature_list=[None]):
        # Gets requested microtiming features or all (if feature_list==[None])
        #
        self.laidbackness = (self.laidback_events - self.pushed_events) if "laidbackness" in feature_list or feature_list == [None] else None
        self.timing_accuracy = self.get_timing_accuracy() if "timing_accuracy" in feature_list or feature_list == [None] else None

    def get_all_features(self):
        # todo: doesn't return triplet-ness
        return np.hstack([self.is_swung, self.swingness, self.laidbackness, self.timing_accuracy])

    def print_all_features(self):
        print("\n   Microtiming features:")
        print("Swingness = " + str(self.swingness))
        print("Is swung = " + str(self.is_swung))
        print("Laidback-ness  = " + str(self.laidbackness))
        print("Timing Accuracy  = " + str(self.timing_accuracy))

    def check_if_swung(self):
        # Check if loop is swung - return 'true' or 'false'

        if self.swingness > 0.0:
            self.is_swung = 1
        elif self.swingness == 0.0:
            self.is_swung = 0

        return self.is_swung

    def get_swing_ratio(self):
        # todo: implement
        pass

    def get_swingness(self):
        return self.swingness

    def get_laidbackness(self):
        self.laidbackness = self.laidback_events - self.pushed_events
        return self.laidbackness

    def _get_swing_info(self):
        # Calculate all of the swing characteristics (swing ratio, swingness etc) in one go

        swung_note_positions = list(range(self.average_timing_matrix.shape[0]))[3::4]

        swing_count = 0.0
        j = 0
        for i in swung_note_positions:
            if self.average_timing_matrix[i] < -25.0:
                swing_count += 1
            j += 1

        swing_count = np.clip(swing_count, 0, len(swung_note_positions))

        if swing_count > 0:
            self.swingness = (1 + (swing_count / len(swung_note_positions)/10))  # todo: weight swing count
        else:
            self.swingness = 0.0

    def _getmicrotiming_event_profile_1bar(self, microtiming_matrix):
        # Get profile of microtiming events for use in pushness/laidbackness/ontopness features
        # This profile represents the presence of specific timing events at certain positions in the pattern
        # Microtiming events fall within the following categories:
        #   Kick timing deviation - before/after metronome, before/after hihat, beats 1 and 3
        #   Snare timing deviation - before/after metronome, before/after hihat, beats 2 and 4
        # As such for one bar the profile contains 16 values.
        # The profile uses binary values - it only measures the presence of timing events, and the style features are
        # then calculated based on the number of events present that correspond to a certain timing feel.

        timing_to_grid_profile = np.zeros([8])
        timing_to_cymbal_profile = np.zeros([8])
        threshold = 12.0
        kick_timing_1 = microtiming_matrix[0, 0]
        hihat_timing_1 = microtiming_matrix[0, 2]
        snare_timing2 = microtiming_matrix[4, 1]
        hihat_timing_2 = microtiming_matrix[4, 2]
        kick_timing_3 = microtiming_matrix[8, 0]
        hihat_timing_3 = microtiming_matrix[8, 2]
        snare_timing4 = microtiming_matrix[12, 1]
        hihat_timing_4 = microtiming_matrix[12, 2]

        if kick_timing_1 > threshold:
            timing_to_grid_profile[0] = 1
        if kick_timing_1 < -threshold:
            timing_to_grid_profile[1] = 1
        if snare_timing2 > threshold:
            timing_to_grid_profile[2] = 1
        if snare_timing2 < -threshold:
            timing_to_grid_profile[3] = 1

        if kick_timing_3 > threshold:
            timing_to_grid_profile[4] = 1
        if kick_timing_3 < -threshold:
            timing_to_grid_profile[5] = 1
        if snare_timing4 > threshold:
            timing_to_grid_profile[6] = 1
        if snare_timing4 < -threshold:
            timing_to_grid_profile[7] = 1

        if kick_timing_1 > hihat_timing_1 + threshold:
            timing_to_cymbal_profile[0] = 1
        if kick_timing_1 < hihat_timing_1 - threshold:
            timing_to_cymbal_profile[1] = 1
        if snare_timing2 > hihat_timing_2 + threshold:
            timing_to_cymbal_profile[2] = 1
        if snare_timing2 < hihat_timing_2 - threshold:
            timing_to_cymbal_profile[3] = 1

        if kick_timing_3 > hihat_timing_3 + threshold:
            timing_to_cymbal_profile[4] = 1
        if kick_timing_3 < hihat_timing_3 - threshold:
            timing_to_cymbal_profile[5] = 1
        if snare_timing4 > hihat_timing_4 + threshold:
            timing_to_cymbal_profile[6] = 1
        if snare_timing4 < hihat_timing_4 - threshold:
            timing_to_cymbal_profile[7] = 1

        microtiming_event_profile_1bar = np.clip(timing_to_grid_profile+timing_to_cymbal_profile, 0, 1)

        return microtiming_event_profile_1bar

    def _get_pushed_events(self):
        # Calculate how 'pushed' the loop is, based on number of pushed events / number of possible pushed events

        push_events = self.microtiming_event_profile[1::2]
        push_event_count = np.count_nonzero(push_events)
        total_push_positions = push_events.shape[0]
        self.pushed_events = push_event_count / total_push_positions
        return self.pushed_events

    def _get_laidback_events(self):
        # Calculate how 'laid-back' the loop is,
        # based on the number of laid back events / number of possible laid back events

        laidback_events = self.microtiming_event_profile[0::2]
        laidback_event_count = np.count_nonzero(laidback_events)
        total_laidback_positions = laidback_events.shape[0]
        self.laidback_events = laidback_event_count / float(total_laidback_positions)
        return self.laidback_events

    def get_timing_accuracy(self):
        # Calculate timing accuracy of the loop

        swung_note_positions = list(range(self.average_timing_matrix.shape[0]))[3::4]
        nonswing_timing = 0.0
        nonswing_note_count = 0
        triplet_positions = 1, 5, 9, 13, 17, 21, 25, 29

        for i in range(self.average_timing_matrix.shape[0]):
            if i not in swung_note_positions and i not in triplet_positions:
                if ~np.isnan(self.average_timing_matrix[i]):
                    nonswing_timing += abs(np.nan_to_num(self.average_timing_matrix[i]))
                    nonswing_note_count += 1
        self.timing_accuracy = nonswing_timing / float(nonswing_note_count)

        return self.timing_accuracy

    def get_average_timing_deviation(self):
        # Get vector of average microtiming deviation at each metrical position

        self.average_timing_matrix = np.zeros([self.microtiming_matrix.shape[0]])
        for i in range(self.microtiming_matrix.shape[0]):
            row_sum = 0.0
            hit_count = 0.0
            row_is_empty = np.all(np.isnan(self.microtiming_matrix[i, :]))
            if row_is_empty:
                self.average_timing_matrix[i] = np.nan
            else:
                for j in range(self.microtiming_matrix.shape[1]):
                    if np.isnan(self.microtiming_matrix[i, j]):
                        pass
                    else:
                        row_sum += self.microtiming_matrix[i, j]
                        hit_count += 1.0
                self.average_timing_matrix[i] = row_sum / hit_count
        return self.average_timing_matrix
