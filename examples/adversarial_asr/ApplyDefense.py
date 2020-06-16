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
import apply_defense_mod
import copy


# data directory
#flags.DEFINE_string("root_dir", "./", "location of Librispeech")
flags.DEFINE_string('input', 'read_data.txt',
                    'Input audio .wav file(s), at 16KHz (separated by spaces)')

# data processing
#flags.DEFINE_integer('window_size', '2048', 'window size in spectrum analysis')
#flags.DEFINE_integer('max_length_dataset', '223200','the length of the longest audio in the whole dataset')
#flags.DEFINE_float('initial_bound', '2000', 'initial l infinity norm for adversarial perturbation')

# training parameters
flags.DEFINE_string('checkpoint', "./model/ckpt-00908156",
                    'location of checkpoint')
flags.DEFINE_integer('batch_size', '5', 'batch size')
flags.DEFINE_boolean('adv', 'False', 'to test adversarial examples or clean examples')

#flags.DEFINE_float('lr_stage1', '100', 'learning_rate for stage 1')
#flags.DEFINE_float('lr_stage2', '1', 'learning_rate for stage 2')
#flags.DEFINE_integer('num_iter_stage1', '1000', 'number of iterations in stage 1')
#flags.DEFINE_integer('num_iter_stage2', '4000', 'number of iterations in stage 2')
#flags.DEFINE_integer('num_gpu', '0', 'which gpu to run')

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

    # read the .wav file
    for i in range(batch_size):
        sample_rate_np, audio_temp = wav.read(FLAGS.root_dir + str(data[0, i]))

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

    return audios_np, trans, th_batch, psd_max_batch, max_length, sample_rate_np, masks, masks_freq, lengths


def applyDefense(batch_size, th_batch, audios_stft):
    noisy = []
    # noisy = [[[0]*1025]*305]*batch_size
    #  for i in range(batch_size):
    for i in range(batch_size):
        temp1 = []
        for j in range(len(th_batch[i])):
            temp2 = []
            for k in range(len(th_batch[i][j])):
                sd = 0 #th_batch[i][j][k]/6  # changed
                mean = 0
                temp2.append(min(max(np.random.normal(mean, sd, 1)[0], 0), th_batch[i][j][k]))

            temp1.append(temp2)
        noisy.append(temp1)
    return noisy


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


'''
def FeedForward (audios, sample_rate, mask_freq): #not finished
    pass_in = tf.clip_by_value(audios, -2 ** 15, 2 ** 15 - 1)
    features = create_features(pass_in, sample_rate, mask_freq) #I think we need to modify create_features method
    inputs = create_inputs(model, features, self.tgt_tf, self.batch_size, self.mask_freq)
'''


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

def overlawAudio(file1, file2, final_file_name):
    sound1 = AudioSegment.from_file(file1)
    sound2 = AudioSegment.from_file(file2)

    combined = sound1.overlay(sound2)

    combined.export(final_file_name, format='wav')
    return 'finished'

def main(argv):
    apply_defense_mod.save_audios(-0.75)
    '''
    data = np.loadtxt(FLAGS.input, dtype=str, delimiter=",")
    data = data[:, FLAGS.num_gpu * 10: (FLAGS.num_gpu + 1) * 10]
    num = len(data[0])
    batch_size = FLAGS.batch_size
    num_loops = round(num / batch_size)
    assert num % batch_size == 0

    num_loops = 1
    for l in range(num_loops):
        for x in range(2): # apply to defense to both benign (1) and adv example (0)

            data_sub = data[:, l * batch_size:(l + 1) * batch_size]
            data_new = copy.deepcopy(data_sub)

            if x == 0:
                for m in range(batch_size):
                    data_new[0][m] = data_sub[0][m][0:len(data_sub[0][m])-4] + '_stage2' + '.wav'



            audios, trans, th_batch, psd_max_batch, maxlen, sample_rate, masks, masks_freq, lengths = ReadFromWav(data_new, batch_size)
            psd_threshold = thresholdPSD(batch_size, th_batch, audios, window_size=2048)

            audio_stft = []
            for i in range(batch_size):
                audio_stft.append(numpy.transpose(abs(librosa.core.stft(audios[i], center=False))))
            noisy = applyDefense(batch_size, psd_threshold, audio_stft)

            for k in range(batch_size):
                phase = []
                phase = ((numpy.angle(librosa.core.stft(audios[k], center=False))))

                time_series = librosa.core.istft(np.array(getPhase(np.transpose(noisy[k]),phase)),center=False)
                #print(numpy.array(time_series, dtype=float))
                #wav.write('defensive_perturbation.wav', sample_rate,numpy.array(time_series, dtype=float))

                time_series1 = librosa.core.istft(np.array(getPhase(np.transpose(audio_stft[k]),phase)),center=False)
                #wav.write('original.wav', sample_rate,numpy.array(time_series1, dtype=float))

                final_time_series = time_series1 + time_series
                final_np = np.array(final_time_series, dtype='int16')

                final_time_series = final_time_series/ 32768.
                #final_np_2 = numpy.array(final_time_series, dtype='int16')
                final_time_series = final_time_series[:lengths[k]]
                final_np_2 = np.copy(final_time_series)
                final_np_2 = final_np_2.astype('float32')

                name = ''
                saved_name = ''
                saved_name_2 = ''
                if x == 0:
                    name, _ = data_sub[0, k].split(".")
                    saved_name = FLAGS.root_dir + str(name) + "_defense.wav"
                    saved_name_2 = FLAGS.root_dir + str(name) + "_defense_2.wav"
                else:
                    name, _ = data_sub[0, k].split(".")
                    saved_name = FLAGS.root_dir + str(name) + "_benign.wav"
                    saved_name_2 = FLAGS.root_dir + str(name) + "_benign_2.wav"


                print(saved_name)
                #overlawAudio('original.wav','defensive_perturbation.wav', saved_name)
                wav.write(saved_name, sample_rate,final_time_series)
                wav.write(saved_name_2, sample_rate,final_np_2)

                sam, audiofile = wav.read(saved_name)
                sam1, audiofile2 = wav.read(FLAGS.root_dir + str(name)+"_stage2.wav")
                print('hello')
    '''
    return 0




if __name__ == '__main__':
    app.run(main)


