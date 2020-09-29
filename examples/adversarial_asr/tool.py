from tensorflow.python import pywrap_tensorflow
import numpy as np
import tensorflow as tf
from lingvo.core import asr_frontend
from lingvo.core import py_utils
import librosa as librosa

def _MakeLogMel(audio, sample_rate):
  audio = tf.expand_dims(audio, axis=0)
  static_sample_rate = 16000
  mel_frontend = _CreateAsrFrontend()
  with tf.control_dependencies(
      [tf.assert_equal(sample_rate, static_sample_rate)]):
    log_mel, _ = mel_frontend.FPropDefaultTheta(audio)
  return log_mel

def _CreateAsrFrontend():
  p = asr_frontend.MelFrontend.Params()
  p.sample_rate = 16000.
  p.frame_size_ms = 25.
  p.frame_step_ms = 10.
  p.num_bins = 80
  p.lower_edge_hertz = 125.
  p.upper_edge_hertz = 7600.
  p.preemph = 0.97
  p.noise_scale = 0.
  p.pad_end = False
  # Stack 3 frames and sub-sample by a factor of 3.
  p.left_context = 2
  p.output_stride = 3
  return p.cls(p)

def create_features(input_tf, sample_rate_tf, mask_freq):
    """
    Return:
        A tensor of features with size (batch_size, max_time_steps, 80)
    """

    features_list = []
    # unstact the features with placeholder
    input_unpack = tf.unstack(input_tf, axis=0)
    for i in range(len(input_unpack)):
        features = _MakeLogMel(input_unpack[i], sample_rate_tf)
        features = tf.reshape(features, shape=[-1, 80])
        features = tf.expand_dims(features, dim=0)
        features_list.append(features)
    features_tf = tf.concat(features_list, axis=0)
    features_tf = features_tf * mask_freq
    return features_tf

def create_inputs(model, features, tgt, batch_size, mask_freq):
    tgt_ids, tgt_labels, tgt_paddings = model.GetTask().input_generator.StringsToIds(tgt)

    # we expect src_inputs to be of shape [batch_size, num_frames, feature_dim, channels]
    src_paddings = tf.zeros([tf.shape(features)[0], tf.shape(features)[1]], dtype=tf.float32)
    src_paddings = 1. - mask_freq[:,:,0]
    src_frames = tf.expand_dims(features, dim=-1)

    inputs = py_utils.NestedMap()
    inputs.tgt = py_utils.NestedMap(
        ids=tgt_ids,
        labels=tgt_labels,
        paddings=tgt_paddings,
        weights=1.0 - tgt_paddings)
    inputs.src = py_utils.NestedMap(src_inputs=src_frames, paddings=src_paddings)
    inputs.sample_ids = tf.zeros([batch_size])
    return inputs

def create_speech_rir(audios, rir, lengths_audios, max_len, batch_size):
    """
    Returns:
        A tensor of speech with reverberations (Convolve the audio with the rir)
    """
    speech_rir = []

    for i in range(batch_size):
        s1 = lengths_audios[i]
        s2 = tf.convert_to_tensor(tf.shape(rir))
        shape = s1 + s2 - 1

        # Compute convolution in fourier space
        sp1 = tf.spectral.rfft(rir, shape)
        sp2 = tf.spectral.rfft(tf.slice(tf.reshape(audios[i], [-1,]), [0], [lengths_audios[i]]), shape)
        ret = tf.spectral.irfft(sp1 * sp2, shape)

        # normalization
        ret /= tf.reduce_max(tf.abs(ret))
        ret *= 2 ** (16 - 1) -1
        ret = tf.clip_by_value(ret, -2 **(16 - 1), 2**(16-1) - 1)
        ret = tf.pad(ret, tf.constant([[0, 100000]]))
        ret = ret[:max_len]

        speech_rir.append(tf.expand_dims(ret, axis=0))
    speech_rirs = tf.concat(speech_rir, axis=0)
    return speech_rirs

def thresholdPSD(batch_size, th_batch, audios, window_size, psd_max):
    psd_threshold_batch = []
    for i in range(batch_size):
        win = np.sqrt(8.0 / 3.) * librosa.core.stft(audios[i], center=False)
        z = abs(win / window_size)
        psd_max = np.max(z * z)

        psd_threshold = 3.0 / 8. * float(window_size) * np.sqrt(
            np.multiply(th_batch[i], psd_max) / float(pow(10, 9.6)))
        psd_threshold_batch.append(psd_threshold)
    return psd_threshold_batch

def get_clipped_sample(psd_max_ori, th, apply_delta, batch_size,window_size):
    transform = Transform(window_size)
    for i in range(batch_size):
        transformed = transform((apply_delta[i, :]), (psd_max_ori)[i])
        x = transform.expandAudios(tf.math.minimum(th, transformed), psd_max_ori)
        if i == 0:
            audios = tf.copy(x)
        else:
            audios = tf.concat([audios, x], 0)
    print(audios)
    return audios
class Transform(object):
    '''
    Return: PSD
    '''
    def __init__(self, window_size):
        self.scale = 8. / 3.
        self.frame_length = int(window_size)
        self.frame_step = int(window_size//4)
        self.window_size = window_size

    def __call__(self, x, psd_max_ori):
        win = tf.contrib.signal.stft(x, self.frame_length, self.frame_step)
        z = self.scale *tf.abs(win / self.window_size)
        psd = tf.square(z)
        PSD = tf.pow(10., 9.6) / tf.reshape(psd_max_ori, [-1, 1, 1]) * psd
        return PSD

    def expandAudios(self, PSD, psd_max_ori):
        psd = PSD*tf.reshape(psd_max_ori, [-1, 1, 1])/tf.pow(10., 9.6)
        z = tf.math.sqrt(psd)
        win = z*self.window_size/self.scale
        x = tf.contrib.signal.inverse_stft(
            win, self.frame_length, self.frame_step,
            window_fn=tf.contrib.signal.inverse_stft_window_fn(self.frame_step))
        return x