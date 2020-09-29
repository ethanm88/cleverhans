'''
New File:
- save_audios method called during execution of the evaluate_defensive_mod.py
- Generates the input for Lingvo Model during testing
- Must first generate defensive peturbation files using write_defensive_delta.py
- Todo: add batch testing

'''
import librosa as librosa
# import tensorflow as tf
# from lingvo import model_imports
# from lingvo import model_registry
import numpy
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import generate_masking_threshold as generate_mask
from tool import Transform, create_features, create_inputs, thresholdPSD
import time
# from lingvo.core import cluster_factory
from absl import flags
from absl import app
import scipy
import random
import pickle
from pydub import AudioSegment
import copy

# data directory
flags.DEFINE_string("root_dir", "./", "location of Librispeech")

# data processing
flags.DEFINE_integer('window_size', '2048', 'window size in spectrum analysis')
flags.DEFINE_integer('max_length_dataset', '223200',
                     'the length of the longest audio in the whole dataset')
flags.DEFINE_float('initial_bound', '2000', 'initial l infinity norm for adversarial perturbation')

# training parameters


flags.DEFINE_float('lr_stage1', '100', 'learning_rate for stage 1')
flags.DEFINE_float('lr_stage2', '1', 'learning_rate for stage 2')
flags.DEFINE_integer('num_iter_stage1', '1000', 'number of iterations in stage 1')
flags.DEFINE_integer('num_iter_stage2', '4000', 'number of iterations in stage 2')
flags.DEFINE_integer('num_gpu', '0', 'which gpu to run')

flags.DEFINE_integer('type_defense', '2', 'type of defense to run ')  # 0: ours, 1: MP3, 2: Quantization

flags.DEFINE_string('perturb_name', '_adaptive_stage2_perturb', 'location of perturbation')


FLAGS = flags.FLAGS


def ReadFromWav(data, batch_size):
    """
    Returns:
        audios_np: a numpy array of size (batch_size, max_length) in float
        trans: a numpy array includes the targeted transcriptions (batch_size, )
        th_batch: a numpy array of the masking threshold, each of size (?, 1025)
        psd_max_batch: a numpy array of the psd_max of the original audio (batch_size)
        max_length: the max length of the batch of audios
        sample_rate_np: a numpy array
        masks: a numpy array of size (batch_size, max_length)
        masks_freq: a numpy array of size (batch_size, max_length_freq, 80)
        lengths: a list of the length of original audios
    """
    audios = []
    lengths = []
    th_batch = []
    psd_max_batch = []
    raw_audio = []

    # read the .wav file
    for i in range(batch_size):
        sample_rate_np, audio_temp = wav.read(FLAGS.root_dir + str(data[0, i]))
        raw_audio.append(audio_temp)
        # read the wav form range from [-32767, 32768] or [-1, 1]
        if max(audio_temp) < 1:
            audio_np = audio_temp * 32768
        else:
            audio_np = audio_temp

        length = len(audio_np)

        audios.append(audio_np)
        lengths.append(length)

    max_length = max(lengths)

    # pad the input audio
    audios_np = np.zeros([batch_size, max_length])
    masks = np.zeros([batch_size, max_length])
    lengths_freq = (np.array(lengths) // 2 + 1) // 240 * 3
    max_length_freq = max(lengths_freq)
    masks_freq = np.zeros([batch_size, max_length_freq, 80])
    for i in range(batch_size):
        audio_float = audios[i].astype(float)
        audios_np[i, :lengths[i]] = audio_float
        masks[i, :lengths[i]] = 1
        masks_freq[i, :lengths_freq[i], :] = 1

        # compute the masking threshold
        th, psd_max = generate_mask.generate_th(audios_np[i], sample_rate_np, FLAGS.window_size)
        th_batch.append(th)
        psd_max_batch.append(psd_max)

    th_batch = np.array(th_batch)
    psd_max_batch = np.array(psd_max_batch)

    # read the transcription
    trans = data[2, :]

    return raw_audio, audios_np, trans, th_batch, psd_max_batch, max_length, sample_rate_np, masks, masks_freq, lengths


def applyDefense(batch_size, th_batch, audios_stft, factor):
    noisy = []
    # noisy = [[[0]*1025]*305]*batch_size
    #  for i in range(batch_size):
    factor = float(factor)
    actual_fac = float(pow(10.0, factor))
    for i in range(batch_size):
        temp1 = []
        for j in range(len(th_batch[i])):
            temp2 = []
            for k in range(len(th_batch[i][j])):
                sd = th_batch[i][j][k] * actual_fac  # changed
                mean = th_batch[i][j][k] * actual_fac * 3
                # temp2.append(min(max(np.random.normal(mean, sd, 1)[0], 0), th_batch[i][j][k])) max was th
                temp2.append((max(np.random.normal(mean, sd, 1)[0], 0)))

            temp1.append(temp2)
        noisy.append(temp1)
    return noisy


def getFreqDomain(batch_size, audios, ATH_batch, sample_rate, th_batch, psd_threshold, num_bins):
    audio_stft = []
    freqs = [[[0] * 1025] * 305] * 5
    for i in range(batch_size):
        audio_stft.append(numpy.transpose(abs(librosa.core.stft(audios[i], center=False))))
        for j in range(num_bins):
            freqs[i][j] = ((np.fft.fftfreq(len(audio_stft[i][j]), d=(1 / sample_rate))))
            # freqs[i][j] = librosa.core.fft_frequencies(sample_rate, len(audio_stft[i][j]))

    noisy = applyDefense(batch_size, psd_threshold, audio_stft)
    for i in range(batch_size):
        ATH_batch[i] = pow(10, ATH_batch[i] / 10.)
        ATH_batch[i] = [x for _, x in sorted(zip(freqs[i][0], ATH_batch[i]))]
        for j in range(num_bins):
            audio_stft[i][j] = [x for _, x in sorted(zip(freqs[i][j], audio_stft[i][j]))]
            th_batch[i][j] = [x for _, x in sorted(zip(freqs[i][j], th_batch[i][j]))]
            psd_threshold[i][j] = [x for _, x in sorted(zip(freqs[i][j], psd_threshold[i][j]))]
            noisy[i][j] = [x for _, x in sorted(zip(freqs[i][j], noisy[i][j]))]
            freqs[i][j].sort()

    return audio_stft, noisy, freqs, th_batch, ATH_batch, psd_threshold


def getPhase(radii, angles):
    return radii * numpy.exp(1j * angles)


def randomPhase(angles):
    randomized_angles = []
    for i in range(len(angles)):
        cur_angles = []
        for j in range(len(angles[0])):
            # x = min(angles[i][j]*random.random()*2,2*np.pi)
            cur_angles.append(2 * np.pi * random.random())
        randomized_angles.append(cur_angles)
    return np.array(randomized_angles)


def overlayAudio(file1, file2, final_file_name):
    sound1 = AudioSegment.from_file(file1)
    sound2 = AudioSegment.from_file(file2)

    combined = sound1.overlay(sound2)

    combined.export(final_file_name, format='wav')
    return 'finished'


def MP3_compression(batch_size, data_new, data_ori):
    all_audios = []
    for i in range(batch_size):
        input_name = FLAGS.root_dir + str(data_new[0, i])
        final_name_mp3 = FLAGS.root_dir + data_ori[0][i][0:len(data_ori[0][i]) - 4] + '_compressed.mp3'
        final_name_wav = FLAGS.root_dir + data_ori[0][i][0:len(data_ori[0][i]) - 4] + '_compressed.wav'

        AudioSegment.from_file(input_name).export(final_name_mp3, format="mp3")
        sound = AudioSegment.from_mp3(final_name_mp3)
        sound.export(final_name_wav, format="wav")
        sample_rate_np, audio_temp = wav.read(str(final_name_wav))

        # read the wav form range from [-32767, 32768] or [-1, 1]
        if max(audio_temp) < 1:
            audio_np = audio_temp * 32768
        else:
            audio_np = audio_temp
        all_audios.append(audio_np)
    all_audios = numpy.array([numpy.array(i) for i in all_audios])
    return all_audios


def quantization(batch_size, audios, q, lengths):
    final_audios = []
    for i in range(batch_size):
        temp_audio = []
        for j in range(len(audios[i])):
            nearest_multiple = q * round(audios[i][j] / q)
            temp_audio.append(nearest_multiple)
        temp_audio = temp_audio[: lengths[i]]
        final_audios.append(temp_audio)
    final_audios_np = numpy.array([numpy.array(i) for i in final_audios])
    final_raw_audio = final_audios_np / 32768.
    return final_raw_audio


def normalize_input(all_time_series, batch_size, lengths):
    for i in range(batch_size):
        if max(all_time_series[i]) < 1:
            all_time_series[i] = all_time_series[i] * 32768
        else:
            all_time_series[i] = all_time_series[i]

    max_length = max(lengths)
    audios_np = np.zeros([batch_size, max_length])
    for i in range(batch_size):
        audio_float = all_time_series[i].astype(float)
        audios_np[i, :lengths[i]] = audio_float
    return audios_np


def initial_audio(batch_size, th_batch, audios):
    # calculate normalized threshold
    psd_threshold = thresholdPSD(batch_size, th_batch, audios, window_size=2048)

    # apply stft to data
    phase = []
    for i in range(batch_size):
        phase = ((np.angle(librosa.core.stft(audios[i], center=False))))
    return psd_threshold, phase


def apply_defensive_perturbation(batch_size, psd_threshold, factor, lengths, raw_audio, phase):
    noisy = []
    # noisy = [[[0]*1025]*305]*batch_size
    #  for i in range(batch_size):

    # generate defensive perturbation
    factor = float(factor)
    actual_fac = float(pow(10.0, factor))
    for i in range(batch_size):
        temp1 = []
        for j in range(len(psd_threshold[i])):
            temp2 = []
            for k in range(len(psd_threshold[i][j])):
                sd = psd_threshold[i][j][k] * actual_fac  # changed
                mean = psd_threshold[i][j][k] * actual_fac * 3
                # temp2.append(min(max(np.random.normal(mean, sd, 1)[0], 0), th_batch[i][j][k])) max was th
                temp2.append((max(np.random.normal(mean, sd, 1)[0], 0)))

            temp1.append(temp2)
        noisy.append(temp1)

    # add defensive perturbation to raw audio
    all_time_series = []
    for k in range(batch_size):
        time_series_noisy = librosa.core.istft(np.array(getPhase(np.transpose(noisy[k]), phase)), center=False)
        # time_series = time_series[: lengths[k]]
        time_series_noisy = np.array(time_series_noisy)
        time_series_noisy.resize(lengths[k], refcheck=False)

        time_series_original = raw_audio[k]
        time_series_original = time_series_original[: lengths[k]]
        time_series_original = np.array(time_series_original)

        final_time_series = np.array(time_series_original + time_series_noisy)
        all_time_series.append(final_time_series.tolist())
    all_time_series = np.array([np.array(i) for i in all_time_series])
    # return normalize_input(all_time_series,batch_size, lengths)
    return all_time_series

# need to setup batch testing

def save_audios(factor, index_loop):
    data = np.loadtxt(FLAGS.input, dtype=str, delimiter=",")
    data = data[:, FLAGS.num_gpu * 10: (FLAGS.num_gpu + 1) * 10]
    num = len(data[0])
    batch_size = FLAGS.batch_size
    num_loops = round(num / batch_size)
    assert num % batch_size == 0

    print(num_loops, num)
    benign_time_series = []
    adv_time_series = []
    num_loops = 1
    for l in range(int(num_loops)):
        l = index_loop

        data_sub = data[:, l * batch_size:(l + 1) * batch_size]
        data_new = copy.deepcopy(data_sub)
        raw_audio, audios, trans, th_batch, psd_max_batch, maxlen, sample_rate, masks, masks_freq, lengths = ReadFromWav(
            data_new, batch_size)
        delta_np = numpy.array([0])
        if FLAGS.adv:
            for m in range(batch_size):
                name = data_sub[0][m][0:len(data_sub[0][m]) - 4]
                perturb_name = name + "_" + FLAGS.perturb_name + '.wav'
                sample_rate_np, delta = wav.read(perturb_name)
                _, audio_orig = wav.read("./" + str(name) + ".wav")
                if max(delta) < 1:
                    delta = delta * 32768
                audio_np = audio_orig + delta
                delta_np = delta
                combined_adv = audio_np / 32768.
                wav.write(name + '_adaptive_combined.wav', 16000,
                          np.array(np.clip(combined_adv[:lengths[m]], -2 ** 15, 2 ** 15 - 1)))
                data_new[0][m] = name + '_adaptive_combined.wav'
                print(name + '_adaptive_combined.wav')

        raw_audio, audios, trans, th_batch, psd_max_batch, maxlen, sample_rate, masks, masks_freq, lengths = ReadFromWav(
            data_new, batch_size)

        # types of defenses
        if FLAGS.type_defense == 2:
            print('Type: Quant')
            defense_time_series = np.array(quantization(batch_size, audios, 256., lengths))
            if FLAGS.adv:
                adv_time_series = defense_time_series
            else:
                benign_time_series = defense_time_series
            return adv_time_series, benign_time_series

        if FLAGS.type_defense == 1:
            print('Type: MP3')
            defense_time_series = np.array(MP3_compression(batch_size, data_new, data_sub))
            if FLAGS.adv:
                adv_time_series = defense_time_series
            else:
                benign_time_series = defense_time_series
            return adv_time_series, benign_time_series

        if FLAGS.type_defense == 0:
            print('Type: Ours')

            '''
            Read random noisy defensive perturbation sample for testing:
            You must run write_defensive_delta.py first (you can stop execution prematurely and edit the code below
            based on the testing batch)
            '''
            file_name = './noisy_data/defensive_' + str(0) + '.pkl'
            pkl_file = open(file_name, 'rb')
            all_noisy = pickle.load(pkl_file)
            pkl_file.close()
            key = str(random.randint(0, 99)) + '_' + str(int(0)) + '_' + str(0)
            noisy_sample = (all_noisy[key])

            defense_time_series = noisy_sample + delta_np

            if FLAGS.adv:
                adv_time_series = defense_time_series
            else:
                benign_time_series = noisy_sample
            return adv_time_series, benign_time_series


    return adv_time_series, benign_time_series




