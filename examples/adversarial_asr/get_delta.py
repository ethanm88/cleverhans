import pickle
import numpy

file_name = 'final_adpative.pkl'
pkl_file = open(file_name, 'rb')
adv_example = pickle.load(pkl_file)
print('Type', type(adv_example))
print(numpy.array(adv_example).shape)
print(adv_example)
pkl_file.close()