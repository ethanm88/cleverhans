'''
New File:
- Graphs CE Loss, Threshold Loss, and values of the adaptive factor from data stored in text files
- Must run an adversarial example generation method to generate data
'''
import matplotlib.pyplot as plt
import numpy as np

def readfromfile(file_name, size):
    all_param = []
    with open(file_name) as input_file:
        for line in input_file:
            line = line.strip()
            for number in line.split():
                all_param.append(float(number))

    return all_param


def main():

    loss_ce_stage1 = readfromfile("loss_ce_stage1.txt", 1)
    loss_ce_robust = readfromfile("loss_ce_robust.txt", 1)
    loss_ce_stage2 = readfromfile("loss_ce_stage2.txt", 1)

    iters = range(0,2000, 10)
    iters2 = range(0,1000, 10)
    iters3 = range(0,4000, 10)
    plt.title('Cross Entropy Loss (Adaptive Attacks)',fontsize=18)
    plt.plot(iters, loss_ce_stage1[-200:], label='Stage 1 (Correctness)')
    plt.plot(iters2, loss_ce_robust, label='Stage 1 (Robustness)')
    plt.plot(iters3, loss_ce_stage2, label='Stage 2')
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('CE Loss', fontsize=14)
    plt.legend()
    plt.show()



    alpha_adapt = readfromfile("alpha_adaptive_7000.txt", 1)
    loss_th_adapt = readfromfile("loss_th_adaptive_7000.txt", 1)
    #loss_th_adapt = np.log(loss_th_adapt)

    alpha_normal = readfromfile("alpha_normal_lr_constant.txt", 1)
    loss_th_normal = readfromfile("loss_th_normal_lr_constant.txt", 1)
    #loss_th_normal = np.log(loss_th_normal)
    iters = range(0,4000, 10)
    #iters2 = range(0,6000, 10)

    plt.title('Alphas (Adaptive vs. Normal)',fontsize=18)
    plt.plot(iters, alpha_adapt[-400:], label='Adaptive Attack')
    plt.plot(iters, alpha_normal[-400:], label='Normal Attack')
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Alpha', fontsize=14)
    plt.legend()
    plt.show()

    plt.title('Threshold Loss (Adaptive vs. Normal)', fontsize=18)
    plt.plot(iters, loss_th_adapt[-400:], label='Adaptive Attack')
    plt.plot(iters, loss_th_normal[-400:], label='Normal Attack')

    plt.legend()
    print(plt.gca().get_ylim(), plt.gca().get_xlim())
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Threshold Loss', fontsize=14)
    #plt.ylabel('WER (%)', fontsize=14)
    plt.show()




main()