import pickle

import matplotlib.pyplot as plt
import pprint

import tensorflow as tf
tf.enable_eager_execution()
import os
from lingvo import model_imports
from lingvo import model_registry
import wer_calculation
import numpy as np
import scipy.io.wavfile as wav
import generate_masking_threshold as generate_mask
from tool import Transform, create_features, create_inputs
import time
from lingvo.core import cluster_factory
from absl import flags
from absl import app
import librosa
import random
from google.colab import files

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
flags.DEFINE_integer('batch_size', '1', 'batch size')
flags.DEFINE_float('lr_stage1', '100', 'learning_rate for stage 1')
flags.DEFINE_float('lr_stage1_robust', '5', 'learning_rate for stage 1_robust')
flags.DEFINE_float('lr_stage2', '1', 'learning_rate for stage 2')
flags.DEFINE_integer('num_iter_stage1', '200', 'number of iterations in stage 1')
flags.DEFINE_integer('num_iter_stage1_robust', '200', 'number of iterations in stage 1_robust')
flags.DEFINE_integer('num_iter_stage2', '4000', 'number of iterations in stage 2')
flags.DEFINE_integer('num_gpu', '0', 'which gpu to run')
flags.DEFINE_float('factor', '-0.75', 'log of defensive perturbation proportionality factor k')

flags.DEFINE_integer('num_counter', '2', 'the initial number of required successful noise samples')
flags.DEFINE_integer('num_goal', '10', 'the initial number of noise samples')
flags.DEFINE_integer('max_delta', '100', 'the max delta added to the max l infinity norm')

flags.DEFINE_integer('num_imperceptible_test', '10', 'number of noise samples to test')
flags.DEFINE_integer('num_imperceptible_pass', '6', 'number of noise samples to pass')
flags.DEFINE_integer('maxlen', '146800', 'maxlen - local')

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

    # read the transcription (changed - now actual labels are the transcriptions)
    trans = data[1, :]

    return raw_audio, audios_np, trans, th_batch, psd_max_batch, max_length, sample_rate_np, masks, masks_freq, lengths


def getPhase(radii, angles):
    return radii * np.exp(1j * angles)


def thresholdPSD(batch_size, th_batch, audios, window_size):
    psd_threshold_batch = []
    for i in range(batch_size):
        #th_batch[i] = np.copy(th_batch[i]).resize(len(audios[i]))
        win = np.sqrt(8.0 / 3.) * librosa.core.stft(audios[i], center=False)
        z = abs(win / window_size)
        psd_max = np.max(z * z)

        #th_batch[i] = np.copy(th_batch[i])
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
    return normalize_input(all_time_series, batch_size, lengths)


def read_noisy(num_loop, batch_size, num_iter_batch):  # only works one adv examples at a time - modify
    # open relevant file for 100 iterations
    file_name = './noisy_data/defensive_' + str(int(num_iter_batch)) + '.pkl'
    pkl_file = open(file_name, 'rb')
    all_noisy = pickle.load(pkl_file)
    pkl_file.close()

    # format noise
    noisy_audios = []

    for i in range(100):
        single_audios = []
        for j in range(batch_size):  # only works for batch_size = 0
            key = str(i) + '_' + str(int(num_loop)) + '_' + str(j)
            single_audios.append((all_noisy[key]).tolist())
        single_audios = np.array([np.array(i) for i in single_audios])
        noisy_audios.append(single_audios)
    noisy_audios = np.array([np.array(i) for i in noisy_audios])
    print('Iter_batch: ', num_iter_batch)
    print('File:', file_name)
    return noisy_audios


class Attack:
    def __init__(self, sess, batch_size=1,
                 lr_stage1=100, lr_stage2=1, num_iter_stage1=200, num_iter_stage1_robust=200, num_iter_stage2=4000,
                 th=None,
                 psd_max_ori=None):

        self.sess = sess
        self.num_iter_stage1 = num_iter_stage1
        self.num_iter_stage1_robust = num_iter_stage1_robust
        self.num_iter_stage2 = num_iter_stage2
        self.batch_size = batch_size
        #self.maxlen_int = maxlen_int_1

        self.is_init = True
        # self.lr_stage1 = lr_stage1

        tf.set_random_seed(1234)
        params = model_registry.GetParams('asr.librispeech.Librispeech960Wpm', 'Test')
        params.random_seed = 1234
        params.is_eval = True
        params.cluster.worker.gpus_per_replica = 1
        cluster = cluster_factory.Cluster(params.cluster)
        self.feed_dict = {}
        with cluster, tf.device(cluster.GetPlacer()):
            model = params.cls(params)
            self.delta_large = tf.Variable(np.zeros((batch_size, FLAGS.max_length_dataset), dtype=np.float32),
                                           name='qq_delta')
            #self.delta_large_1 = tf.Variable(self.clip_freq(self.feed_dict),name='qq_delta_1')

            # placeholders
            self.input_tf = tf.placeholder(tf.float32, shape=[batch_size, None], name='qq_input')
            self.ori_input_tf = tf.placeholder(tf.float32, shape=[batch_size, None], name='qq_ori_input')

            self.tgt_tf = tf.placeholder(tf.string)
            self.sample_rate_tf = tf.placeholder(tf.int32, name='qq_sample_rate')
            self.th = tf.placeholder(tf.float32, shape=[batch_size, None, None], name='qq_th')
            self.psd_max_ori = tf.placeholder(tf.float32, shape=[batch_size], name='qq_psd')
            self.mask = tf.placeholder(dtype=np.float32, shape=[batch_size, None], name='qq_mask')
            self.mask_freq = tf.placeholder(dtype=np.float32, shape=[batch_size, None, 80])
            self.noise = tf.placeholder(np.float32, shape=[batch_size, None], name="qq_noise")
            self.maxlen = tf.placeholder(np.int32)
            self.lr_stage2 = tf.placeholder(np.float32)
            self.lr_stage1 = tf.placeholder(np.float32)
            # variable
            self.rescale = tf.Variable(np.ones((batch_size, 1), dtype=np.float32) * FLAGS.initial_bound,
                                       name='qq_rescale')

            self.rescale_th = tf.Variable(np.ones(batch_size, dtype=np.float32) * 0.1,
                                          name='qq_resth')


            self.alpha = tf.Variable(np.ones((batch_size), dtype=np.float32) * 0.001, name='qq_alpha')

            # extract the delta
            #self.apply_delta_th = tf.Variable(np.zeros((batch_size, FLAGS.max_length_dataset), dtype=np.float32),name='qq_apply_delta_th')
            self.delta = tf.slice(tf.identity(self.delta_large), [0, 0], [batch_size, self.maxlen])






            #self.apply_delta_th = tf.Variable(tf.identity(self.delta))


            #self.apply_delta_th = self.clip_freq(place_holder_dict)
            #self.apply_delta_th = (self.clip_freq(self.feed_dict))
            #self.clip_freq(self.feed_dict)




            #self.apply_delta = tf.clip_by_value(self.apply_delta_th, -self.rescale, self.rescale)
            self.apply_delta = tf.clip_by_value(self.delta, -self.rescale, self.rescale)


            self.new_input = self.apply_delta * self.mask + self.input_tf # changed

            self.actual_input = self.apply_delta * self.mask + self.ori_input_tf

            self.pass_in = tf.clip_by_value(self.new_input + self.noise, -2 ** 15, 2 ** 15 - 1)

            # generate the inputs that are needed for the lingvo model
            self.features = create_features(self.pass_in, self.sample_rate_tf, self.mask_freq)
            self.inputs = create_inputs(model, self.features, self.tgt_tf, self.batch_size, self.mask_freq)

            task = model.GetTask()
            metrics = task.FPropDefaultTheta(self.inputs)

            # self.celoss with the shape (batch_size)

            self.celoss = -1 * tf.get_collection("per_loss")[0]  # negate to generate untargeted attacks
            self.decoded = task.Decode(self.inputs)

        # compute the loss for masking threshold
        self.loss_th_list = []
        self.transform = Transform(FLAGS.window_size)
        for i in range(self.batch_size):
            logits_delta = self.transform((self.apply_delta[i, :]), (self.psd_max_ori)[i])
            loss_th = tf.reduce_mean(tf.nn.relu(logits_delta - (self.th)[i]))
            loss_th = tf.expand_dims(loss_th, dim=0)
            self.loss_th_list.append(loss_th)
        self.loss_th = tf.concat(self.loss_th_list, axis=0)

        self.optimizer1 = tf.train.AdamOptimizer(self.lr_stage1)
        self.optimizer2 = tf.train.AdamOptimizer(self.lr_stage2)

        grad1, var1 = self.optimizer1.compute_gradients(self.celoss, [self.delta_large])[0]
        grad21, var21 = self.optimizer2.compute_gradients(self.celoss, [self.delta_large])[0]
        grad22, var22 = self.optimizer2.compute_gradients(self.alpha * self.loss_th, [self.delta_large])[0]

        self.train1 = self.optimizer1.apply_gradients([(tf.sign(grad1), var1)])
        self.train21 = self.optimizer2.apply_gradients([(grad21, var21)])
        self.train22 = self.optimizer2.apply_gradients([(grad22, var22)])
        self.train2 = tf.group(self.train21, self.train22)

    def clip_freq(self, feed_dict):
        print('entered')
        #print(feed_dict)
        if self.is_init == True:
            print("hello")
            return tf.identity(self.delta)
        sess = self.sess
        '''
        #, psd_threshold, delta, batch_size, rescale_th
        original_delta = np.copy((self.delta).numpy())
        batch_size = self.batch_size
        rescale_th = np.copy((self.rescale_th).numpy())

        th_batch = self.th.numpy()

        audios = (self.audios).numpy()
        
        '''

        print('worked!!!')

        original_delta = (sess.run((self.delta_large), feed_dict)).copy()
        maxlen_data_set = sess.run((self.maxlen), feed_dict)
        batch_size = self.batch_size
        rescale_th = np.copy(sess.run((self.rescale_th), feed_dict))
        #print(rescale_th)

        th_batch = np.copy(sess.run((self.th), feed_dict))

        audios = np.copy(sess.run((self.ori_input_tf), feed_dict))


        psd_threshold = thresholdPSD(batch_size, th_batch, audios, window_size=2048)

        clipped_freq = []

        phase = []


        original_delta_np = []
        for i in range(batch_size):
            original_delta_np.append(np.resize((original_delta[i]),(maxlen_data_set)))
            clipped_freq.append(np.transpose(np.abs(librosa.core.stft(original_delta_np[i], center=False))))
            phase = ((np.angle(librosa.core.stft(original_delta_np[i], center=False))))


        for i in range(batch_size):
            for j in range(len(psd_threshold[i])):
                for k in range(len(psd_threshold[i][j])):
                    clipped_freq[i][j][k] = min(clipped_freq[i][j][k], psd_threshold[i][j][k] * rescale_th[i])


        clipped_final = []

        for i in range(batch_size):
            clipped_final.append(np.resize(librosa.core.istft(np.array(getPhase(np.transpose(clipped_freq[i]), phase)), center=False), FLAGS.max_length_dataset))
            #clipped_final[i] = clipped_final[i].resize(FLAGS.max_length_dataset)

        clipped_final = np.array([np.array(i) for i in clipped_final])
        return tf.convert_to_tensor(clipped_final)

    def clip_freq1(self, psd_threshold, delta, batch_size, rescale_th):

        original_delta = np.copy(delta)
        clipped_freq = []

        phase = []

        for i in range(batch_size):

            clipped_freq.append(np.transpose(librosa.core.stft(original_delta[i], center=False)))
            phase = ((np.angle(librosa.core.stft(original_delta[i], center=False))))
        print(self.maxlen)
        print(np.shape(clipped_freq))
        print(np.shape(psd_threshold))
        for i in range(batch_size):
            for j in range(len(psd_threshold[i])):
                for k in range(len(psd_threshold[i][j])):

                    if psd_threshold[i][j][k] * rescale_th[i] < clipped_freq[i][j][k]:
                        print(i,j,k)

                    clipped_freq[i][j][k] = min(clipped_freq[i][j][k], psd_threshold[i][j][k] * rescale_th[i])
        clipped_final = []


        for i in range(batch_size):
            clipped_final.append(librosa.core.istft(np.array(getPhase(np.transpose(clipped_freq[i]), phase)), center=False))

        clipped_final = np.array([np.array(i) for i in clipped_final])
        return clipped_final

    def attack_stage1(self, raw_audio, batch_size, lengths, audios, trans, th_batch, psd_max_batch, maxlen, sample_rate,
                      masks, masks_freq, num_loop,
                      data, lr_stage2, lr_stage1):
        sess = self.sess
        # initialize and load the pretrained model
        sess.run(tf.initializers.global_variables())
        saver = tf.train.Saver([x for x in tf.global_variables() if x.name.startswith("librispeech")])
        saver.restore(sess, FLAGS.checkpoint)

        # reassign the variables
        sess.run(tf.assign(self.rescale, np.ones((self.batch_size, 1), dtype=np.float32) * FLAGS.initial_bound))
        sess.run(tf.assign(self.delta_large, np.zeros((self.batch_size, FLAGS.max_length_dataset), dtype=np.float32)))

        self.is_init = False

        # noise = np.random.normal(scale=2, size=audios.shape)
        noise = np.zeros(audios.shape)

        #psd_threshold, phase = initial_audio(batch_size, th_batch, audios)

        # noisy_audios = apply_defensive_perturbation(batch_size, psd_threshold, FLAGS.factor, lengths, raw_audio, phase)
        noisy_audios = read_noisy(num_loop, batch_size, 0)  # initial noise
        feed_dict = {self.input_tf: noisy_audios[0],
                     self.ori_input_tf: audios,
                     self.tgt_tf: trans,
                     self.sample_rate_tf: sample_rate,
                     self.th: th_batch,
                     self.psd_max_ori: psd_max_batch,
                     self.mask: masks,
                     self.mask_freq: masks_freq,
                     self.noise: noise,
                     self.maxlen: maxlen,
                     self.lr_stage2: lr_stage2,
                     self.lr_stage1: lr_stage1
                     }
        self.feed_dict = feed_dict

        losses, predictions = sess.run((self.celoss, self.decoded), feed_dict)

        # show the initial predictions
        for i in range(self.batch_size):
            print("example: {}, loss: {}".format(num_loop * self.batch_size + i, losses[i]))
            print("pred:{}".format(predictions['topk_decoded'][i, 0]))
            print("targ:{}".format(trans[i].lower()))
            print("true: {}".format(data[1, i].lower()))

        # We'll make a bunch of iterations of gradient descent here
        now = time.time()
        MAX = self.num_iter_stage1
        loss_th = [np.inf] * self.batch_size
        final_deltas = [None] * self.batch_size
        final_perturb = [None] * self.batch_size

        min_difference = 50  # minimum WER required for success

        clock = 0

        for i in range(MAX):
            now = time.time()
            if i % 100 == 0 and i != 0:  # load new file every 100 iterations
                noisy_audios = read_noisy(num_loop, batch_size, int(i / 100))
            feed_dict = {self.input_tf: noisy_audios[i % 100],
                         self.ori_input_tf: audios,
                         self.tgt_tf: trans,
                         self.sample_rate_tf: sample_rate,
                         self.th: th_batch,
                         self.psd_max_ori: psd_max_batch,
                         self.mask: masks,
                         self.mask_freq: masks_freq,
                         self.noise: noise,
                         self.maxlen: maxlen,
                         self.lr_stage2: lr_stage2,
                         self.lr_stage1: lr_stage1
                         }
            self.feed_dict = feed_dict

            # losses, predictions = sess.run((self.celoss, self.decoded), feed_dict)

            # Actually do the optimization
            print('start')
            sess.run(tf.assign(self.delta_large, self.clip_freq(feed_dict)))
            print('end')
            sess.run(self.train1, feed_dict)
            if i % 10 == 0:

                '''
                
                first_delta, d, rescale_th = sess.run((self.apply_delta, self.delta, self.rescale_th), feed_dict)

                freq_clipped_perturb = self.clip_freq(thresholdPSD(batch_size, th_batch, audios, window_size=2048), first_delta, batch_size, rescale_th)
                sess.run(tf.assign(self.apply_delta_th, freq_clipped_perturb))

                apply_delta, cl, predictions, new_input = sess.run(
                    (self.apply_delta_th, self.celoss, self.decoded,
                     self.new_input), feed_dict)
                '''


                loss_th, apply_delta, d, cl, predictions, new_input = sess.run(
                    (self.loss_th, self.apply_delta, self.delta, self.celoss, self.decoded,
                     self.new_input), feed_dict)

                print("Loss th: ", loss_th)

            for ii in range(self.batch_size):
                # print out the prediction each 100 iterations

                if i % 10 == 0:
                    print("Every:")
                    print("iteration_Test: %d" % (i))
                    print("loss_ce_Test: %f" % (cl[ii]))
                    print("Current distortion",
                          np.max(np.abs(new_input[ii][:lengths[ii]] - noisy_audios[i % 100])))

                    with open("loss_ce.txt", "a") as text_file:
                        text_file.write(str(cl[ii]) + "\n")
                    with open("distortion.txt", "a") as text_file:
                        text_file.write(
                            str(np.max(np.abs(new_input[ii][:lengths[ii]] - audios[ii, :lengths[ii]]))) + "\n")

                if i % 50 == 0:
                    print("pred:{}".format(predictions['topk_decoded'][ii, 0]))
                    print("targ:{}".format(trans[ii].lower()))
                    print("true: {}".format(data[1, ii].lower()))
                    # print("rescale: {}".format(sess.run(self.rescale[ii])))
                if i % 10 == 0:
                    if i % 100 == 0:
                        print("example: {}".format(num_loop * self.batch_size + ii))
                        print("iteration: {}. loss {}".format(i, cl[ii]))

                    # if predictions['topk_decoded'][ii, 0] == trans[ii].lower():
                    WER = wer_calculation.wer(trans[ii].lower().split(), predictions['topk_decoded'][ii, 0].split())
                    print("WER: ", WER)
                    if WER >= min_difference:
                        print("-------------------------------True--------------------------")
                        rescale = sess.run(self.rescale)
                        rescale_th = sess.run(self.rescale_th)
                        # update rescale
                        if i % 10 == 0:
                            if rescale[ii] > np.max(np.abs(d[ii])):
                                rescale[ii] = np.max(np.abs(d[ii]))
                            rescale[ii] *= .8
                            rescale_th[ii] *= 0.8

                        # save the best adversarial example
                        final_deltas[ii] = new_input[ii]
                        final_perturb[ii] = apply_delta[ii]

                        print("Iteration i=%d, worked ii=%d celoss=%f bound=%f" % (
                            i, ii, cl[ii], rescale[ii]))
                        print('rescale_th', rescale_th)
                        sess.run(tf.assign(self.rescale, rescale))
                        sess.run(tf.assign(self.rescale_th, rescale_th))

                # in case no final_delta return
                if (i == MAX - 1 and final_deltas[ii] is None):
                    final_deltas[ii] = new_input[ii]
                    final_perturb[ii] = apply_delta[ii]

            if i % 10 == 0:
                print("ten iterations take around {} ".format(clock))
                clock = 0

            clock += time.time() - now

        return final_deltas, final_perturb

    def attack_stage1_robust(self, adv, rescales, raw_audio, batch_size, lengths, audios, trans, th_batch,
                             psd_max_batch, maxlen, sample_rate,
                             masks, masks_freq, num_loop,
                             data, lr_stage2):
        sess = self.sess
        # initialize and load the pretrained model
        sess.run(tf.initializers.global_variables())
        saver = tf.train.Saver([x for x in tf.global_variables() if x.name.startswith("librispeech")])
        saver.restore(sess, FLAGS.checkpoint)

        # reassign the variables
        sess.run(tf.assign(self.delta_large, adv))
        # sess.run(tf.assign(self.lr_stage1, FLAGS.lr_stage1_robust))
        sess.run(tf.assign(self.rescale, rescales))

        noise = np.zeros(audios.shape)

        noisy_audios = read_noisy(num_loop, batch_size, 20)  # initial noise
        noisy_audios_testing = read_noisy(num_loop, batch_size, random.randint(0, 49))  # noise for robustness testing

        feed_dict = {self.input_tf: noisy_audios[0],
                     self.ori_input_tf: audios,
                     self.tgt_tf: trans,
                     self.sample_rate_tf: sample_rate,
                     self.th: th_batch,
                     self.psd_max_ori: psd_max_batch,
                     self.mask: masks,
                     self.mask_freq: masks_freq,
                     self.noise: noise,
                     self.maxlen: maxlen,
                     self.lr_stage2: lr_stage2,
                     self.lr_stage1: FLAGS.lr_stage1_robust
                     }
        losses, predictions = sess.run((self.celoss, self.decoded), feed_dict)

        # show the initial predictions
        for i in range(self.batch_size):
            print("example: {}, loss: {}".format(num_loop * self.batch_size + i, losses[i]))
            print("pred:{}".format(predictions['topk_decoded'][i, 0]))
            print("targ:{}".format(trans[i].lower()))
            print("true: {}".format(data[1, i].lower()))

        # We'll make a bunch of iterations of gradient descent here
        now = time.time()
        MAX = self.num_iter_stage1_robust
        loss_th = [np.inf] * self.batch_size
        final_deltas = [None] * self.batch_size
        final_perturb = [None] * self.batch_size

        min_difference = 50  # minimum WER required for success

        num_counters = [FLAGS.num_counter] * self.batch_size
        num_goal = [FLAGS.num_goal] * self.batch_size

        clock = 0
        cur_file = 20
        print('Num Iters: ', MAX)
        for i in range(MAX):
            now = time.time()

            if i % 100 == 0 and i != 0:  # load new file every 100 iterations
                cur_file = (int(20 + i / 100) % 50)
                noisy_audios = read_noisy(num_loop, batch_size, (int(20 + i / 100) % 50))

            feed_dict = {self.input_tf: noisy_audios[i % 100],
                         self.ori_input_tf: audios,
                         self.tgt_tf: trans,
                         self.sample_rate_tf: sample_rate,
                         self.th: th_batch,
                         self.psd_max_ori: psd_max_batch,
                         self.mask: masks,
                         self.mask_freq: masks_freq,
                         self.noise: noise,
                         self.maxlen: maxlen,
                         self.lr_stage2: lr_stage2,
                         self.lr_stage1: FLAGS.lr_stage1_robust
                         }

            # Actually do the optimization
            sess.run(self.train1, feed_dict)
            if i % 10 == 0:
                apply_delta, d, cl, predictions, new_input = sess.run(
                    (self.apply_delta, self.delta, self.celoss, self.decoded,
                     self.new_input), feed_dict)

            if i % 10 == 0:
                index = random.randint(0, 49)
                while index == cur_file:
                    index = random.randint(0, 49)
                noisy_audios_testing = read_noisy(num_loop, batch_size,
                                                  index)  # get random noise file - move into loop when get better gpu

            for ii in range(self.batch_size):
                # print out the prediction each 100 iterations

                if i % 10 == 0:
                    print("Every:")
                    print("iteration_Test: %d" % (i))
                    print("loss_ce_Test: %f" % (cl[ii]))

                    with open("loss_ce.txt", "a") as text_file:
                        text_file.write(str(cl[ii]) + "\n")

                if i % 50 == 0:
                    print("pred:{}".format(predictions['topk_decoded'][ii, 0]))
                    print("targ:{}".format(trans[ii].lower()))
                    print("true: {}".format(data[1, ii].lower()))
                    print("rescale: {}".format(sess.run(self.rescale[ii])))

                sum_counter = 0
                if i % 10 == 0:
                    for counter in range(num_goal[ii]):
                        WER = wer_calculation.wer(trans[ii].lower().split(), predictions['topk_decoded'][ii, 0].split())
                        print("WER: ", WER)

                        # if predictions['topk_decoded'][ii, 0] == trans[ii].lower():
                        if WER >= min_difference:
                            sum_counter += 1
                            print("succeed %d times for example %d" % (sum_counter, ii))

                            feed_dict = {self.input_tf: noisy_audios_testing[counter % 100],
                                         self.ori_input_tf: audios,
                                         self.tgt_tf: trans,
                                         self.sample_rate_tf: sample_rate,
                                         self.th: th_batch,
                                         self.psd_max_ori: psd_max_batch,
                                         self.mask: masks,
                                         self.mask_freq: masks_freq,
                                         self.noise: noise,
                                         self.maxlen: maxlen,
                                         self.lr_stage2: lr_stage2,
                                         self.lr_stage1: FLAGS.lr_stage1_robust
                                         }
                            predictions = sess.run(self.decoded, feed_dict)
                            if counter % 100 == 99:
                                index = random.randint(0, 49)
                                while index == cur_file:
                                    index = random.randint(0, 49)
                                noisy_audios_testing = read_noisy(num_loop, batch_size, index)

                        if sum_counter == num_counters[ii]:
                            print("-------------------------------True--------------------------")
                            print(" The num_counter is %d for example %d" % (num_counters[ii], ii))
                            num_counters[ii] += 1
                            if num_counters[ii] > num_goal[ii]:
                                num_goal[ii] += 1
                            # save the best adversarial example
                            final_deltas[ii] = new_input[ii]
                            final_perturb[ii] = apply_delta[ii]
                            print("Stage 1_robust: save the example at iteration i=%d example ii=%d celoss=%f" % (
                                i, ii, cl[ii]))

                # in case no final_delta return
                if (i == MAX - 1 and final_deltas[ii] is None):
                    final_deltas[ii] = new_input[ii]
                    final_perturb[ii] = apply_delta[ii]

            if i % 10 == 0:
                print("ten iterations take around {} ".format(clock))
                clock = 0

            clock += time.time() - now

        return final_deltas, final_perturb

    def attack_stage2(self, raw_audio, batch_size, lengths, rescales, audios, trans, adv, th_batch, psd_max_batch,
                      maxlen,
                      sample_rate, masks, masks_freq,
                      num_loop, data, lr_stage2, lr_stage1):
        sess = self.sess
        # initialize and load the pretrained model
        sess.run(tf.initializers.global_variables())
        saver = tf.train.Saver([x for x in tf.global_variables() if x.name.startswith("librispeech")])
        saver.restore(sess, FLAGS.checkpoint)

        sess.run(tf.assign(self.rescale, rescales))
        sess.run(tf.assign(self.alpha, np.ones((self.batch_size), dtype=np.float32) * 0.001))

        # reassign the variables
        sess.run(tf.assign(self.delta_large, adv))

        # noise = np.random.normal(scale=2, size=audios.shape)
        noise = np.zeros(audios.shape)

        psd_threshold, phase = initial_audio(batch_size, th_batch, audios)
        # noisy_audios = apply_defensive_perturbation(batch_size, psd_threshold, FLAGS.factor, lengths, raw_audio, phase)
        noisy_audios = read_noisy(num_loop, batch_size, 40)  # load initial audio
        noisy_audios_testing = read_noisy(num_loop, batch_size, random.randint(0, 49))  # noise for robustness testing

        feed_dict = {self.input_tf: noisy_audios[0],
                     self.ori_input_tf: audios,
                     self.tgt_tf: trans,
                     self.sample_rate_tf: sample_rate,
                     self.th: th_batch,
                     self.psd_max_ori: psd_max_batch,
                     self.mask: masks,
                     self.mask_freq: masks_freq,
                     self.noise: noise,
                     self.maxlen: maxlen,
                     self.lr_stage2: lr_stage2,
                     self.lr_stage1: lr_stage1}
        losses, predictions = sess.run((self.celoss, self.decoded), feed_dict)

        # show the initial predictions
        for i in range(self.batch_size):
            print("example: {}, loss: {}".format(num_loop * self.batch_size + i, losses[i]))
            print("pred:{}".format(predictions['topk_decoded'][i, 0]))
            print("targ:{}".format(trans[i].lower()))
            print("true: {}".format(data[1, i].lower()))

        # We'll make a bunch of iterations of gradient descent here
        now = time.time()
        MAX = self.num_iter_stage2
        loss_th = [np.inf] * self.batch_size
        final_deltas = [None] * self.batch_size
        final_alpha = [None] * self.batch_size
        final_perturb = [None] * self.batch_size
        # final_th = [None] * self.batch_size
        clock = 0
        min_th = 0.000005

        min_difference = 50  # minimum WER required for success

        num_imperceptible_test = [FLAGS.num_imperceptible_test] * self.batch_size
        num_imperceptible_pass = [FLAGS.num_imperceptible_pass] * self.batch_size

        cur_file = 0
        print('Num Iters: ', MAX)
        for i in range(MAX):  # changed - start at 20000

            now = time.time()
            if i % 100 == 0 and i != 0:  # load new file every 100 iterations
                cur_file = int(30 + i / 100) % 50
                noisy_audios = read_noisy(num_loop, batch_size, int(40 + i / 100) % 50)

            # if i%50 == 0 and i!= 0:
            #    noisy_audios_testing = read_noisy(num_loop, batch_size,random.randint(0, 49))  # noise for robustness testing

            feed_dict = {self.input_tf: noisy_audios[i % 100],
                         self.ori_input_tf: audios,
                         self.tgt_tf: trans,
                         self.sample_rate_tf: sample_rate,
                         self.th: th_batch,
                         self.psd_max_ori: psd_max_batch,
                         self.mask: masks,
                         self.mask_freq: masks_freq,
                         self.noise: noise,
                         self.maxlen: maxlen,
                         self.lr_stage2: lr_stage2,
                         self.lr_stage1: lr_stage1}
            # losses, predictions = sess.run((self.celoss, self.decoded), feed_dict)

            '''
            if i == 3000: # 6000 iterations now
                # min_th = -np.inf
                #lr_stage2 = 0.1
                feed_dict = {self.input_tf: noisy_audios[i % 100],
                             self.ori_input_tf: audios,
                             self.tgt_tf: trans,
                             self.sample_rate_tf: sample_rate,
                             self.th: th_batch,
                             self.psd_max_ori: psd_max_batch,
                             self.mask: masks,
                             self.mask_freq: masks_freq,
                             self.noise: noise,
                             self.maxlen: maxlen,
                             self.lr_stage2: lr_stage2,
                             self.lr_stage1: lr_stage1}
            '''

            # Actually do the optimization
            sess.run(self.train2, feed_dict)
            # print('Delta_large', self.delta_large)
            # print('Alpha', self.alpha)
            if i % 10 == 0:
                apply_delta, d, cl, l, predictions, new_input = sess.run(
                    (self.apply_delta, self.delta, self.celoss, self.loss_th, self.decoded, self.new_input), feed_dict)

            if i % 10 == 0:
                if i % 10 == 0:
                    index = random.randint(0, 49)
                    while index == cur_file:
                        index = random.randint(0, 49)
                    noisy_audios_testing = read_noisy(num_loop, batch_size,
                                                      index)  # get random noise file - move into loop when get better gpu

            for ii in range(self.batch_size):
                # print out the prediction each 50 iterations
                if i % 50 == 0:
                    print("pred:{}".format(predictions['topk_decoded'][ii, 0]))
                    print("targ:{}".format(trans[ii].lower()))
                    print("true: {}".format(data[1, ii].lower()))
                    print("rescale: {}".format(sess.run(self.rescale[ii])))
                if i % 10 == 0:
                    # print("example: {}".format(num_loop * self.batch_size + ii))

                    alpha = sess.run(self.alpha)
                    print("Every:")
                    print("iteration_Test: %d" % (i))
                    print("alpha_Test: %f" % (alpha[ii]))
                    print("loss_ce_Test: %f" % (cl[ii]))
                    print("loss_th_Test: %f" % (l[ii]))

                    with open("alpha.txt", "a") as text_file:
                        text_file.write(str(alpha[ii]) + "\n")
                    with open("loss_th.txt", "a") as text_file:
                        text_file.write(str(l[ii]) + "\n")
                    with open("loss_ce_stage2.txt", "a") as text_file:
                        text_file.write(str(cl[ii]) + "\n")

                    if i % 100 == 0:
                        print("example: {}".format(num_loop * self.batch_size + ii))
                        print("iteration: %d, alpha: %f, loss_ce: %f, loss_th: %f" % (i, alpha[ii], cl[ii], l[ii]))

                    # if the network makes the targeted prediction
                    sum_counter = 0
                    sum_succeed = 0
                    for counter in range(num_imperceptible_test[ii]):
                        print('Iter:', counter)
                        WER = wer_calculation.wer(trans[ii].lower().split(), predictions['topk_decoded'][ii, 0].split())
                        print("WER: ", WER)

                        # if predictions['topk_decoded'][ii, 0] == trans[ii].lower():
                        if WER >= min_difference:
                            sum_succeed += 1
                            if l[ii] < loss_th[ii]:
                                sum_counter += 1
                                print("succeed %d times for example %d" % (sum_counter, ii))
                            else:
                                print("succeed at %d but loss_th too high for example %d" % (counter, ii))
                        else:
                            print("fail at %d for example %d" % (counter, ii))

                        noise_index = random.randint(0, 99)
                        feed_dict = {self.input_tf: noisy_audios_testing[noise_index],
                                     self.ori_input_tf: audios,
                                     self.tgt_tf: trans,
                                     self.sample_rate_tf: sample_rate,
                                     self.th: th_batch,
                                     self.psd_max_ori: psd_max_batch,
                                     self.mask: masks,
                                     self.mask_freq: masks_freq,
                                     self.noise: noise,
                                     self.maxlen: maxlen,
                                     self.lr_stage2: lr_stage2,
                                     self.lr_stage1: FLAGS.lr_stage1
                                     }
                        predictions = sess.run(self.decoded, feed_dict)

                        if sum_counter == num_imperceptible_pass[ii]:
                            final_deltas[ii] = new_input[ii]
                            final_perturb[ii] = apply_delta[ii]
                            loss_th[ii] = l[ii]
                            final_alpha[ii] = alpha[ii]
                            print("-------------------------------------Succeed---------------------------------")
                            print("save the best example=%d at iteration= %d, alpha = %f" % (ii, i, alpha[ii]))
                            break

                    if sum_succeed >= num_imperceptible_pass[ii]:
                        # increase the alpha each 20 iterations
                        if i % 20 == 0:
                            alpha[ii] *= 1.2
                            sess.run(tf.assign(self.alpha, alpha))

                    # if the network fails to make the targeted prediction, reduce alpha each 50 iterations
                    if i % 50 == 0 and sum_succeed < num_imperceptible_pass[ii]:
                        alpha[ii] *= 0.8
                        alpha[ii] = max(alpha[ii], min_th)
                        sess.run(tf.assign(self.alpha, alpha))

                # in case no final_delta return
                if (i == MAX - 1 and final_deltas[ii] is None):
                    final_deltas[ii] = new_input[ii]
                    final_perturb[ii] = apply_delta[ii]

            if i % 500 == 0:
                print("alpha is {}, loss_th is {}".format(final_alpha, loss_th))
            if i % 10 == 0:
                print("ten iterations take around {} ".format(clock))
                clock = 0

            clock += time.time() - now

        return final_deltas, final_perturb, loss_th, final_alpha


def main(argv):
    data = np.loadtxt(FLAGS.input, dtype=str, delimiter=",")
    data = data[:, FLAGS.num_gpu * 10: (FLAGS.num_gpu + 1) * 10]
    num = len(data[0])
    batch_size = FLAGS.batch_size
    num_loops = num / batch_size
    assert num % batch_size == 0



    with tf.device("/gpu:0"):  # changed
        tfconf = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=tfconf) as sess:


            # set up the attack class
            attack = Attack(sess,
                            batch_size=batch_size,
                            lr_stage1=FLAGS.lr_stage1,
                            lr_stage2=FLAGS.lr_stage2,
                            num_iter_stage1=FLAGS.num_iter_stage1,
                            num_iter_stage2=FLAGS.num_iter_stage2

                            )
            num_loops = 1
            batch_size = 1
            for l in range(num_loops):

                data_sub = data[:, l * batch_size:(l + 1) * batch_size]

                # stage 1
                # all the output are numpy arrays
                raw_audio, audios, trans, th_batch, psd_max_batch, maxlen, sample_rate, masks, masks_freq, lengths = ReadFromWav(
                    data_sub, batch_size)
                print("maxlen", maxlen)

                adv_example, perturb = attack.attack_stage1(raw_audio, batch_size, lengths, audios, trans, th_batch, psd_max_batch, maxlen, sample_rate, masks,
                                                   masks_freq, l, data_sub, FLAGS.lr_stage2, FLAGS.lr_stage1)



                # save the adversarial examples in stage 1
                for i in range(batch_size):
                    print("Final distortion for stage 1",
                          np.max(np.abs(adv_example[i][:lengths[i]] - audios[i, :lengths[i]])))
                    name, _ = data_sub[0, i].split(".")
                    saved_name = FLAGS.root_dir + str(name) + "_adaptive_untargeted_legal_stage1.wav"
                    adv_example_float = adv_example[i] / 32768.
                    wav.write(saved_name, 16000, np.array(adv_example_float[:lengths[i]]))
                    print(saved_name)

                    saved_name = FLAGS.root_dir + str(name) + "_adaptive_untargeted_legal_stage1_perturb.wav"
                    perturb_float = perturb[i] / 32768.
                    wav.write(saved_name, 16000, np.array(np.clip(perturb_float[:lengths[i]], -2 ** 15, 2 ** 15 - 1)))
                    print(saved_name)


                # stage 1_robust
                # read the adversarial examples saved in stage 1

                #read previous
                raw_audio, audios, trans, th_batch, psd_max_batch, maxlen, sample_rate, masks, masks_freq, lengths = ReadFromWav(data_sub, batch_size)

                for i in range(batch_size):
                    name, _ = data_sub[0, i].split(".")
                    saved_name = FLAGS.root_dir + str(name) + "_adaptive_untargeted_legal_stage1_perturb.wav"
                    sample_rate_np, perturb = wav.read(saved_name)

                    _, audio_orig = wav.read("./" + str(name) + ".wav")

                    if max(perturb) < 1:
                        perturb = perturb * 32768
                    adv_example = audios[i] + perturb

                adv = np.zeros([batch_size, FLAGS.max_length_dataset])
                adv[:, :maxlen] = adv_example - audios
                rescales = np.max(np.abs(adv), axis=1) + FLAGS.max_delta
                rescales = np.expand_dims(rescales, axis=1)

                raw_audio, audios, trans, th_batch, psd_max_batch, maxlen, sample_rate, masks, masks_freq, lengths = ReadFromWav(data_sub, batch_size)
                adv_example, perturb = attack.attack_stage1_robust(adv, rescales, raw_audio, batch_size, lengths, audios, trans, th_batch, psd_max_batch, maxlen, sample_rate, masks,
                                                            masks_freq, l, data_sub, FLAGS.lr_stage2)


                # save the adversarial examples in stage 2 that can successfully attack a set of simulated rooms
                for i in range(batch_size):
                    print("Final distortion for stage 1_robust",
                          np.max(np.abs(adv_example[i][:lengths[i]] - audios[i, :lengths[i]])))
                    name, _ = data_sub[0, i].split(".")
                    saved_name = FLAGS.root_dir + str(name) + "_adaptive_untargeted_legal_stage1_robust.wav"
                    adv_example_float = adv_example[i] / 32768.
                    wav.write(saved_name, 16000, np.array(np.clip(adv_example_float[:lengths[i]], -2 ** 15, 2 ** 15 - 1)))

                    saved_name = FLAGS.root_dir + str(name) + "_adaptive_untargeted_legal_stage1_robust_perturb.wav"
                    perturb_float = perturb[i] / 32768.
                    wav.write(saved_name, 16000, np.array(np.clip(perturb_float[:lengths[i]], -2 ** 15, 2 ** 15 - 1)))
                    print(saved_name)





                '''

                # stage 2
                # read the adversarial examples saved in stage 1

                # read previous
                raw_audio, audios, trans, th_batch, psd_max_batch, maxlen, sample_rate, masks, masks_freq, lengths = ReadFromWav(
                    data_sub, batch_size)

                for i in range(batch_size):
                    name, _ = data_sub[0, i].split(".")
                    saved_name = FLAGS.root_dir + str(name) + "_adaptive_untargeted_legal_stage1_robust_perturb.wav"
                    sample_rate_np, perturb = wav.read(saved_name)

                    _, audio_orig = wav.read("./" + str(name) + ".wav")

                    if max(perturb) < 1:
                        perturb = perturb * 32768
                    adv_example = audios[i] + perturb  # change to audios[i]

                    print(saved_name)

                adv = np.zeros([batch_size, FLAGS.max_length_dataset])
                adv[:, :maxlen] = adv_example - audios
                rescales = np.max(np.abs(adv), axis=1)
                rescales = np.expand_dims(rescales, axis=1)
                adv_example, perturb, loss_th, final_alpha = attack.attack_stage2(raw_audio, batch_size, lengths,
                                                                                  rescales, audios, trans,
                                                                                  adv, th_batch, psd_max_batch,
                                                                                  maxlen, sample_rate, masks,
                                                                                  masks_freq, l,
                                                                                  data_sub, FLAGS.lr_stage2,
                                                                                  FLAGS.lr_stage1)

                print('final_loss_th: ', loss_th)
                # save the adversarial examples in stage 2
                for i in range(batch_size):
                    # save adv examples:

                    print("example: {}".format(i))
                    print("Final distortion for stage 2: {}, final alpha is {}, final loss_th is {}".format(
                        np.max(np.abs(adv_example[i][:lengths[i]] - audios[i, :lengths[i]])), final_alpha[i],
                        loss_th[i]))

                    name, _ = data_sub[0, i].split(".")
                    saved_name = FLAGS.root_dir + str(name) + "_adaptive_untargeted_legal_stage2.wav"
                    adv_example_float = adv_example[i] / 32768.
                    print('size', np.array(adv_example[i][:lengths[i]]).size)

                    wav.write(saved_name, 16000, (np.array(adv_example_float[:lengths[i]])).transpose())
                    print(saved_name)

                    name, _ = data_sub[0, i].split(".")
                    saved_name = FLAGS.root_dir + str(name) + "_adaptive_untargeted_legal_stage2_perturb.wav"
                    perturb_float = perturb[i] / 32768.
                    print('size', np.array(adv_example[i][:lengths[i]]).size)

                    wav.write(saved_name, 16000, (np.array(perturb_float[:lengths[i]])).transpose())
                    print(saved_name)
                '''

if __name__ == '__main__':
    app.run(main)