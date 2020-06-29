import pickle

file_name = 'adaptive_stage_1.pkl'
pkl_file = open(file_name, 'rb')
adv_example = pickle.load(pkl_file)
print('Type', type(adv_example))
print(adv_example)
pkl_file.close()