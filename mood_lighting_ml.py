import numpy
from pyAudioAnalysis.MidTermFeatures import mid_feature_extraction as mF
from pyAudioAnalysis import audioTrainTest as aT
import datetime
import signal
import struct
import color_map_2d
import audioop

global fs
global all_data
global outstr
fs = 8000

"""
Load 2D image of the valence-arousal representation and define coordinates
of emotions and respective colors
"""
img = (622, 622, 0)

"""
Color definition and emotion colormap definition
"""
colors = {
          "orange": [255, 165, 0],
          "blue": [0, 0, 255],
          "bluegreen": [0, 165, 255],
          "green": [0, 205, 0],
          "red": [255, 0, 0],
          "yellow": [255, 255, 0],
          "purple": [128, 0, 128],
          "neutral": [255, 241, 224]}

disgust_pos = [-0.9, 0]
angry_pos = [-0.5, 0.5]
alert_pos = [0, 0.6]
happy_pos = [0.5, 0.5]
calm_pos = [0.4, -0.4]
relaxed_pos = [0, -0.6]
sad_pos = [-0.5, -0.5]
neu_pos = [0.0, 0.0]

# Each point in the valence/energy map is represented with a static color based
# on the above mapping. All intermediate points of the emotion colormap
# are then computed using the color_map_2d.create_2d_color_map() function:

emo_map = color_map_2d.create_2d_color_map([disgust_pos,
                                            angry_pos,
                                            alert_pos,
                                            happy_pos,
                                            calm_pos,
                                            relaxed_pos,
                                            sad_pos,
                                            neu_pos],
                                           [colors["purple"],
                                            colors["red"],
                                            colors["orange"],
                                            colors["yellow"],
                                            colors["green"],
                                            colors["bluegreen"],
                                            colors["blue"],
                                            colors["neutral"]],
                                           img[0], img[1])


def get_color_from_audio(block, rms_min_max=[0, 25000]):
    mid_buf = []
    global all_data
    global outstr
    all_data = []
    outstr = datetime.datetime.now().strftime("%Y_%m_%d_%I:%M%p")

    # load segment model
    [classifier, mu, std, class_names,
     mt_win, mt_step, st_win, st_step, _] = aT.load_model("model")

    [clf_energy, mu_energy, std_energy, class_names_energy,
     mt_win_en, mt_step_en, st_win_en, st_step_en, _] = \
        aT.load_model("energy")

    [clf_valence, mu_valence, std_valence, class_names_valence,
     mt_win_va, mt_step_va, st_win_va, st_step_va, _] = \
        aT.load_model("valence")

    count_b = len(block) / 2
    format_h = "%dh" % (count_b)
    shorts = struct.unpack(format_h, block)
    cur_win = list(shorts)
    mid_buf = mid_buf + cur_win
    del cur_win
    if len(mid_buf) >= 5 * fs:
        # data-driven time
        x = numpy.int16(mid_buf)
        seg_len = len(x)
        r = audioop.rms(x, 2)
        if r < rms_min_max[0]:
            # set new min incase the default value is exceded
            rms_min_max[0] = r
        if r > rms_min_max[1]:
            # set new max incase the default value is exceded
            rms_min_max[1] = r
        r_norm = float(r - rms_min_max[0])/float(rms_min_max[1] - rms_min_max[0])
        r_map = int(r_norm * 255)
        print(f'RMS: {r}; MIN: {rms_min_max[0]}; MAX: {rms_min_max[1]}; NORM: {r_norm}; MAP: {r_map}')
        # extract features
        # We are using the signal length as mid term window and step,
        # in order to guarantee a mid-term feature sequence of len 1
        [mt_f, _, _] = mF(x, fs, seg_len, seg_len, round(fs * st_win),
                          round(fs * st_step))
        fv = (mt_f[:, 0] - mu) / std
        # classify vector:
        [res, prob] = aT.classifier_wrapper(classifier, "svm_rbf", fv)
        win_class = class_names[int(res)]
        if prob[class_names.index("silence")] > 0.8:
            soft_valence = 0
            soft_energy = 0
            print("Silence")
        else:
            # extract features for music mood
            [f_2, _, _] = mF(x, fs, round(fs * mt_win_en),
                             round(fs * mt_step_en), round(fs * st_win_en),
                             round(fs * st_step_en))
            [f_3, _, _] = mF(x, fs, round(fs * mt_win_va),
                             round(fs * mt_step_va), round(fs * st_win_va),
                             round(fs * st_step_va))

            # normalize feature vector
            fv_2 = (f_2[:, 0] - mu_energy) / std_energy
            fv_3 = (f_3[:, 0] - mu_valence) / std_valence
            [res_energy, p_en] = aT.classifier_wrapper(clf_energy,
                                                       "svm_rbf",
                                                       fv_2)

            win_class_energy = class_names_energy[int(res_energy)]
            [res_valence, p_val] = aT.classifier_wrapper(clf_valence,
                                                         "svm_rbf",
                                                         fv_3)

            win_class_valence = class_names_valence[int(res_valence)]
            soft_energy = p_en[class_names_energy.index("high")] - \
                          p_en[class_names_energy.index("low")]
            soft_valence = p_val[class_names_valence.index("positive")] - \
                           p_val[class_names_valence.index("negative")]

            print(win_class, win_class_energy, win_class_valence,
                  soft_valence, soft_energy)

        all_data += mid_buf
        mid_buf = []
        h, w, _ = img
        y_center, x_center = int(h / 2), int(w / 2)
        x = x_center + int((w/2) * soft_valence)
        y = y_center - int((h/2) * soft_energy)

        radius = 20
        color = numpy.median(emo_map[y-2:y+2, x-2:x+2], axis=0).mean(axis=0)
        # set sisyphus led colors
        alpha = format(r_map, '02x')
        return format(int(color[2]), '02x') + format(int(color[1]), '02x') + format(int(color[0]), '02x') + alpha
