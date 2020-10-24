'''
New File
    - displays graph of WERs with respect to log (k) -> data is incorrect (generate new data)
'''

import matplotlib.pyplot as plt

def readfromfile(file_name, size):
    all_WER = []
    with open(file_name) as input_file:
        for line in input_file:
            line = line.strip()
            for number in line.split():
                all_WER.append(float(number))
    avg_WER = get_averages(size, all_WER)
    return avg_WER

def get_averages(size, list):
    avg_WER = []
    for i in range(0, len(list), 3):
        avg_WER.append((list[i]+list[i+1]+list[i+2])/3.0)
    return avg_WER

def main():
    set_2_adv = readfromfile("./Results/results_adv-5-9.txt", 3)
    set_2_benign = readfromfile("./Results/results_benign-5-9.txt", 3)
    set_2_revert = readfromfile("./Results/results_revert-5-9.txt", 3)

    k = []
    for i in range(-900, 125, 25):
        k.append(i/100)

    all_adv = []
    all_benign = []
    all_revert = []
    min_revert_rate = 0
    index = 0
    plt.plot(k,set_2_adv, label = 'Adversarial to Adversarial WER')
    plt.plot(k, set_2_benign, label='Benign to Benign WER')
    plt.plot(k, set_2_revert, label='Adversarial to Benign WER')
    plt.legend()
    print(plt.gca().get_ylim(), plt.gca().get_xlim())
    plt.xlabel('log(k)', fontsize=14)
    plt.ylabel('WER (%)', fontsize=14)

    plt.show()






main()