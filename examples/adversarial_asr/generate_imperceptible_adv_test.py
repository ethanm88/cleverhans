import librosa as librosa
#import tensorflow as tf
#from lingvo import model_imports
#from lingvo import model_registry
import numpy
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import generate_masking_threshold as generate_mask
#from tool import Transform, create_features, create_inputs
import time
#from lingvo.core import cluster_factory
from absl import flags
from absl import app
import scipy

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

FLAGS = flags.FLAGS


def ReadFromWav(data, batch_size):
    """
    Returns: 
        audios_np: a numpy array of size (batch_size, max_length) in float, For each sample, there is an list  of amplitudes with values between -32768 and +32768
        trans: a numpy array includes the targeted transcriptions (batch_size, )
        th_batch: a numpy array of the masking threshold, each of size (?, 1025). Sample * frame * bin within frame
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

    ATH_batch = []

    for i in range(batch_size):
        audio_float = audios[i].astype(float)
        audios_np[i, :lengths[i]] = audio_float
        masks[i, :lengths[i]] = 1
        masks_freq[i, :lengths_freq[i], :] = 1
 
        # compute the masking threshold        
        th, psd_max,ATH = generate_mask.generate_th(audios_np[i], sample_rate_np, FLAGS.window_size)
        th_batch.append(th)
        psd_max_batch.append(psd_max)
        ATH_batch.append(ATH)
     
    th_batch = np.array(th_batch)
    psd_max_batch = np.array(psd_max_batch)
    
    # read the transcription
    trans = data[2, :]
    
    return audios_np, trans, th_batch, psd_max_batch, max_length, sample_rate_np, masks, masks_freq, lengths, ATH_batch
        


def applyDefense (batch_size, th_batch, audios_stft):
    noisy = []
    #noisy = [[[0]*1025]*305]*batch_size
  #  for i in range(batch_size):
    for i in range(batch_size):
        temp1 = []
        for j in range(len(th_batch[i])):
            temp2 = []
            for k in range(len(th_batch[i][j])):

                sd = th_batch[i][j][k]/20 # changed
                if th_batch[i][j][k]>audios_stft[i][j][k]:
                    temp2.append(min(max(audios_stft[i][j][k] + np.random.normal(0, sd, 1)[0], 0),th_batch[i][j][k]))
                else:
                    temp2.append(max(audios_stft[i][j][k] + np.random.normal(0, sd, 1)[0], 0))
                #defensive_perturb = np.random.normal(0, sd, 1)[0]
                #temp2.append(max(audios_stft[i][j][k] + np.random.normal(0, sd, 1)[0],0))

            temp1.append(temp2)
        noisy.append(temp1)
    return noisy

def applyPartialDefense (sample_num, th_batch, audios_stft):
    noisy = []
    #noisy = [[[0]*1025]*305]*batch_size
  #  for i in range(batch_size):


    for j in range(len(th_batch[sample_num])):
        temp2 = []
        for k in range(len(th_batch[sample_num][j])):
            if(th_batch[sample_num][j][k]>150000 and (i == 0)):
                print(sample_num,j,k)
            sd = 2*th_batch[sample_num][j][k]/3
            defensive_perturb = np.random.normal(0, sd, 1)[0]
            temp2.append(max(audios_stft[sample_num][j][k] + np.random.normal(0, sd, 1)[0],0))

        noisy.append(temp2)
    return noisy

def graphs(audio_stft, noisy, freqs, th_batch_sorted, ATH_batch, sample_num, bin_num):



    axes = plt.gca()
    axes.set_xlim([20, 8000])
    plt.plot(freqs[sample_num][bin_num], ATH_batch[sample_num], label='Threshold in Quiet')
    plt.plot(freqs[sample_num][bin_num], noisy[sample_num][bin_num], label = 'Original Audio + Defensive Perturbation')
    plt.plot(freqs[sample_num][bin_num], audio_stft[sample_num][bin_num], label = 'Raw Audio')
    plt.plot(freqs[sample_num][bin_num], th_batch_sorted[sample_num][bin_num], label = 'Masking Threshold')
    plt.legend()
    plt.xlabel('frequency (hz)', fontsize=10)
    plt.ylabel('amplitude', fontsize=10)
    #plt.legend(handles=[line_one, line_two, line_three])
    plt.show()


def getFreqDomain(batch_size, audios, ATH_batch, sample_rate, th_batch, num_bins):
    audio_stft = []
    freqs = [[[0]*1025]*305]*5
    for i in range(batch_size):
        audio_stft.append(numpy.transpose(abs(librosa.core.stft(audios[i], center=False))))
        for j in range(num_bins):
            freqs[i][j] = ((np.fft.fftfreq(len(audio_stft[i][j]), d=(1 / sample_rate))))
            #freqs[i][j] = librosa.core.fft_frequencies(sample_rate, len(audio_stft[i][j]))

    noisy = applyDefense(batch_size, th_batch, audio_stft)
    for i in range(batch_size):
        ATH_batch[i] = pow(10, ATH_batch[i] / 10.)
        ATH_batch[i] = [x for _, x in sorted(zip(freqs[i][0], ATH_batch[i]))]
        for j in range(num_bins):
            audio_stft[i][j] = [x for _, x in sorted(zip(freqs[i][j], audio_stft[i][j]))]
            th_batch[i][j] = [x for _, x in sorted(zip(freqs[i][j], th_batch[i][j]))]
            noisy[i][j] = [x for _, x in sorted(zip(freqs[i][j], noisy[i][j]))]
            freqs[i][j].sort()

    return audio_stft, noisy, freqs, th_batch, ATH_batch
'''
def FeedForward (audios, sample_rate, mask_freq): #not finished
    pass_in = tf.clip_by_value(audios, -2 ** 15, 2 ** 15 - 1)
    features = create_features(pass_in, sample_rate, mask_freq) #I think we need to modify create_features method
    inputs = create_inputs(model, features, self.tgt_tf, self.batch_size, self.mask_freq)
'''

def getPhase(radii, angles):
    return radii * numpy.exp(1j * angles)


def main(argv):
    data = np.loadtxt(FLAGS.input, dtype=str, delimiter=",")
    data = data[:, FLAGS.num_gpu * 10 : (FLAGS.num_gpu + 1) * 10]
    num = len(data[0])
    batch_size = FLAGS.batch_size
    num_loops = round(num / batch_size)
    assert num % batch_size == 0
    for l in range(num_loops):
        data_sub = data[:, l * batch_size:(l + 1) * batch_size]
        audios, trans, th_batch, psd_max_batch, maxlen, sample_rate, masks, masks_freq, lengths, ATH_batch = ReadFromWav(data_sub,batch_size)
        #audio_stft, noisy, freqs, th_batch_sorted, ATH_batch = getFreqDomain(batch_size, audios, ATH_batch, sample_rate, th_batch, 305) #not always 305
        #graphs(audio_stft, noisy, freqs, th_batch_sorted, ATH_batch, 0, 0)

        if(l == 0):
            audio_stft = []
            phase = []
            for i in range(batch_size):
                audio_stft.append(numpy.transpose(abs(librosa.core.stft(audios[i], center=False))))
                phase.append((numpy.angle(librosa.core.stft(audios[i], center=False))))

            noisy = applyDefense(batch_size,th_batch, audio_stft)
            totalATH = [pow(10, ATH_batch[i] / 10.)]*len(audio_stft[0])

            time_series_th = librosa.core.istft(np.array(getPhase(np.transpose(totalATH),phase[0])),center=False)
            wav.write('threshold.wav', sample_rate,numpy.array(time_series_th, dtype='int16'))

            time_series = librosa.core.istft(np.array(getPhase(np.transpose(noisy[0]),phase[0])),center=False)
            wav.write('test1.wav', sample_rate,numpy.array(time_series, dtype='int16'))

            time_series1 = librosa.core.istft(np.array(getPhase(np.transpose(audio_stft[0]),phase[0])),center=False)
            wav.write('original.wav', sample_rate,numpy.array(time_series1, dtype='int16'))

            wav.write('original1.wav', sample_rate, numpy.array(audios[0], dtype='int16'))
        # convert magnitude back to phase

        '''
        full_masking_threshold = []
        
        audio_stft = get_freq_domain(batch_size, audios)
        freqs = np.fft.fftfreq(len(audio_stft[0]), d=(1/sample_rate))
        audio_stft[1] = [x for _, x in sorted(zip(freqs, audio_stft[1]))]
        th_batch[0][1] = [x for _, x in sorted(zip(freqs, th_batch[0][1]))]

        noisy = ApplyDefense(batch_size, th_batch, audio_stft)
        noisy[1] = [x for _, x in sorted(zip(freqs, noisy[1]))]
        freqs.sort()
        
        
        axes = plt.gca()
        axes.set_xlim([1, 8000])
        plt.plot(freqs, noisy[1])
        plt.plot(freqs, audio_stft[1])


        #plt.plot(freqs, audio_stft[1])
        #plt.plot(freqs, th_batch[0][1])
        plt.show()

        print(noisy[1])


        #print(len(audio_stft[1][1]))



        full_audio = []
        '''
        '''
        for i in range(len(th_batch[0]-1)):
            for j in range(len(th_batch[0][i]-1)):
                full_audio.append(audio_stft[j][i])
                full_masking_threshold.append(th_batch[0][i][j])

        print('audio', full_audio)

        plt.plot(full_masking_threshold)
        plt.show()
        print('masking', full_masking_threshold)
        plt.plot(full_audio)
        plt.show()
'''
    '''
    with tf.device("/gpu:0"):
        tfconf = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=tfconf) as sess: 
            # set up the attack class
            attack = Attack(sess, 
                            batch_size=batch_size,
                            lr_stage1=FLAGS.lr_stage1,
                            lr_stage2=FLAGS.lr_stage2,
                            num_iter_stage1=FLAGS.num_iter_stage1,
                            num_iter_stage2=FLAGS.num_iter_stage2)

            for l in range(num_loops):
                data_sub = data[:, l * batch_size:(l + 1) * batch_size] 
                               
                # stage 1
                # all the output are numpy arrays
                audios, trans, th_batch, psd_max_batch, maxlen, sample_rate, masks, masks_freq, lengths = ReadFromWav(data_sub, batch_size)                                                                      
                adv_example = attack.attack_stage1(audios, trans, th_batch, psd_max_batch, maxlen, sample_rate, masks, masks_freq, l, data_sub, FLAGS.lr_stage2)
                
                # save the adversarial examples in stage 1
                for i in range(batch_size):
                    print("Final distortion for stage 1", np.max(np.abs(adv_example[i][:lengths[i]] - audios[i, :lengths[i]])))                                      
                    name, _ = data_sub[0, i].split(".")                    
                    saved_name = FLAGS.root_dir + str(name) + "_stage1.wav"                     
                    adv_example_float =  adv_example[i] / 32768.
                    wav.write(saved_name, 16000, np.array(adv_example_float[:lengths[i]]))
                    print(saved_name)                    
                                    
                # stage 2                
                # read the adversarial examples saved in stage 1
                adv = np.zeros([batch_size, FLAGS.max_length_dataset])
                adv[:, :maxlen] = adv_example - audios

                adv_example, loss_th, final_alpha = attack.attack_stage2(audios, trans, adv, th_batch, psd_max_batch, maxlen, sample_rate, masks, masks_freq, l, data_sub, FLAGS.lr_stage2)
                
                # save the adversarial examples in stage 2
                for i in range(batch_size):
                    print("example: {}".format(i))                    
                    print("Final distortion for stage 2: {}, final alpha is {}, final loss_th is {}".format(np.max(np.abs(adv_example[i][:lengths[i]] - audios[i, :lengths[i]])), final_alpha[i], loss_th[i]))
                                     
                    name, _ = data_sub[0, i].split(".")                 
                    saved_name = FLAGS.root_dir + str(name) + "_stage2.wav"                                       
                    adv_example[i] =  adv_example[i] / 32768. #this line should be removed
                    wav.write(saved_name, 16000, np.array(adv_example[i][:lengths[i]]))
                    print(saved_name)                    
                    
                #Experiment 1
                adv_example_noised = ApplyDefense(adv_example, th_batch, batch_size)
                #run adv_example_noised and adv_example through classifier
                #need add training component
                #Experiment 2

                #Experiment 3
                '''
if __name__ == '__main__':
    app.run(main)
    
    
