import librosa as librosa
# import tensorflow as tf
# from lingvo import model_imports
# from lingvo import model_registry
import numpy
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import generate_masking_threshold as generate_mask
# from tool import Transform, create_features, create_inputs
import time
# from lingvo.core import cluster_factory
from absl import flags
from absl import app
import scipy
import random
from pydub import AudioSegment
import copy
import pickle
import pprint


# data directory
flags.DEFINE_string("root_dir", "./", "location of Librispeech")
flags.DEFINE_string('input', 'read_data.txt',
                    'Input audio .wav file(s), at 16KHz (separated by spaces)')

# data processing
flags.DEFINE_integer('window_size', '2048', 'window size in spectrum analysis')
flags.DEFINE_integer('max_length_dataset', '223200',
                     'the length of the longest audio in the whole dataset')
flags.DEFINE_float('initial_bound', '2000', 'initial l infinity norm for adversarial perturbation')

# training parameters
flags.DEFINE_string('checkpoint', "./model/ckpt-00908156",
                    'location of checkpoint')
flags.DEFINE_integer('batch_size', '5', 'batch size')
flags.DEFINE_float('lr_stage1', '100', 'learning_rate for stage 1')
flags.DEFINE_float('lr_stage2', '1', 'learning_rate for stage 2')
flags.DEFINE_integer('num_iter_stage1', '1000', 'number of iterations in stage 1')
flags.DEFINE_integer('num_iter_stage2', '4000', 'number of iterations in stage 2')
flags.DEFINE_integer('num_gpu', '0', 'which gpu to run')
flags.DEFINE_float('factor', '-0.75', 'log of defensive perturbation proportionality factor k')

FLAGS = flags.FLAGS




from absl import app

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

def getPhase(radii, angles):
    return radii * np.exp(1j * angles)

def thresholdPSD(batch_size, th_batch, audios, window_size):
    psd_threshold_batch = []
    for i in range(batch_size):
        win = np.sqrt(8.0 / 3.) * librosa.core.stft(audios[i], center=False)
        z = abs(win / window_size)
        psd_max = np.max(z * z)

        psd_threshold = np.sqrt(3.0 / 8.) * float(window_size) * np.sqrt(
            np.multiply(th_batch[i], psd_max) / float(pow(10, 9.6)))
        psd_threshold_batch.append(psd_threshold)
    return psd_threshold_batch

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

    #generate defensive perturbation
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

    #add defensive perturbation to raw audio
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
    return normalize_input(all_time_series,batch_size, lengths)

def main(argv):
    data = np.loadtxt(FLAGS.input, dtype=str, delimiter=",")
    data = data[:, FLAGS.num_gpu * 10: (FLAGS.num_gpu + 1) * 10]
    num = len(data[0])
    batch_size = FLAGS.batch_size
    num_loops = num / batch_size

    assert num % batch_size == 0

    all_noisy_data = {}
    num_loops = 1
    for l in range(int(num_loops)):
        data_sub = data[:, l * batch_size:(l + 1) * batch_size]
        raw_audio, audios, trans, th_batch, psd_max_batch, maxlen, sample_rate, masks, masks_freq, lengths = ReadFromWav(data_sub, batch_size)
        psd_threshold, phase = initial_audio(batch_size, th_batch, audios)

    # save noisy files 5000 of each for each iteration
        num_total_iter = 5000
        for i in range(num_total_iter):
            print("Iter: ", i)
            noisy_audios = apply_defensive_perturbation(batch_size, psd_threshold, FLAGS.factor, lengths, raw_audio,
                                                        phase)
            for j in range(batch_size):
                print("Iter: ", i, "Audio Number: ", j)
                key = str(i) + '_' + str(l) + '_' + str(j)
                all_noisy_data.update({key:noisy_audios[j]})

    output = open('defensive.pkl', 'wb')
    pickle.dump(all_noisy_data, output)
    output.close()

    pkl_file = open('defensive.pkl', 'rb')

    data1 = pickle.load(pkl_file)
    pprint.pprint(data1)

    pkl_file.close()

    '''
    my_data = {'a': [[1,2,4],[4,5,6]],
               'b': ('string', u'Unicode string'),
               'c': None}
    output = open('data.pkl', 'wb')
    pickle.dump(my_data, output)
    output.close()

    pkl_file = open('data.pkl', 'rb')

    data1 = pickle.load(pkl_file)
    pprint.pprint(data1['a'])

    pkl_file.close()
    '''








if __name__ == '__main__':
    app.run(main)


