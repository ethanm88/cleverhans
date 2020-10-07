'''
Delete/change slightly
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
    #y = [0.00, 84.33734939759, 91.0843373494, 100.24096385556, 104.09638554207, 107.10843373516, 105.6626506024,111.9277108434, 104.81927710847, 107.22891566279, 103.61445783146, 106.3855421688, 110.24096385559, 109.75903614459, 105.5421686747, 107.590361446, 104.8192771085, 102.8915662651, 101.80722891569, 101.68674698799, 100.843373494, 102.04819277109, 99.87951807229, 99.79919678715]
    #x = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23]
    benign = []
    for i in range(29):
        benign.append(4.38596491228)
    add = [5.26315789474, 5.26315789474, 5.26315789474, 7.01754385965, 7.01754385965, 7.01754385965, 7.60233918129, 7.8947368421067, 16.959064327467, 18.421052631567, 23.0994152047, 27.192982456133]
    benign = benign+add
    adv = [0, 0, 14.05622489957, 20.883534136533, 21.285140562267, 21.285140562267, 30.522088353433, 38.955823293167, 50.200803212833, 71.4859437751, 84.7389558233, 95.180722891567, 101.60642570267, 108.03212851411, 107.09504685422, 100, 99.866131191433, 99.5983935743, 100, 99.5983935743, 100, 100, 100, 100, 100, 94.7791164659, 104.41767068233, 131.72690763067, 128.514056225, 132.530120482, 132.12851405633, 132.12851405633, 131.325301205, 132.530120482, 132.12851405633, 130.92369477933, 132.530120482, 132.530120482, 137.34939759033, 138.554216867, 138.15261044167]

    revert = [96.4912280702, 96.4912280702, 96.4912280702, 96.4912280702, 96.4912280702, 96.4912280702, 96.4912280702, 96.198830409333, 96.198830409333, 97.660818713433, 95.0292397661, 99.122807017533, 95.029239766067, 93.567251462, 92.397660818733, 97.953216374267, 99.707602339167, 99.415204678333, 100, 99.707602339167, 100, 100, 100, 100, 99.415204678333, 92.982456140333, 82.748538011667, 40.6432748538, 33.0409356725, 23.099415204667, 23.391812865533, 20.4678362573, 17.836257309967, 19.298245614033, 19.298245614033, 20.4678362573, 21.9298245614, 26.9005847953, 31.2865497076, 37.7192982456, 57.017543859667]

    k = [-9, -8.75, -8.5, -8.25, -8, -7.75, -7.5, -7.25, -7.0, -6.75, -6.5, -6.25, -6.00, -5.75, -5.50, -5.25, -5.00, -4.75, -4.5, -4.25, -4.00, -3.75, -3.50, -3.25, -3.00, -2.75, -2.50, -2.25, -2.00, -1.75, -1.50, -1.25, -1.00, -0.75, -0.5, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00]
    for i in range(len(adv)):
        print(k[i], revert[i])
    all_adv = []
    all_benign = []
    all_revert = []
    min_revert_rate = 0
    index = 0

    print('Revert:')
    for i in range(len(benign)):
        all_adv.append((adv[i] + set_2_adv[i])/2.0)
        all_benign.append((benign[i] + set_2_benign[i])/2.0)
        all_revert.append((revert[i] + set_2_revert[i])/2.0)
        print(k[i], all_adv[i],all_benign[i], all_revert[i])
        if min_revert_rate<all_adv[i]:
            min_revert_rate=all_adv[i]
            index = i
    print('revert_rate',index, min_revert_rate)


    plt.plot(k,all_adv, label = 'Adversarial to Adversarial WER')
    plt.plot(k, all_benign, label='Benign to Benign WER')
    plt.plot(k, all_revert, label='Adversarial to Benign WER')
    plt.legend()
    print(plt.gca().get_ylim(), plt.gca().get_xlim())
    plt.xlabel('log(k)', fontsize=14)
    plt.ylabel('WER (%)', fontsize=14)
    #plt.ylim([-6.927710843350001, 145.48192771035002])
    #plt.xlim([-9.5, 1.5])

    plt.show()




main()