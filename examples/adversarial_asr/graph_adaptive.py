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
    alpha_adapt = readfromfile("alpha_adaptive.txt", 1)
    loss_th_adapt = readfromfile("loss_th_adaptive.txt", 1)
    loss_th_adapt = np.log(loss_th_adapt)

    alpha_normal = readfromfile("alpha_normal.txt", 1)
    loss_th_normal = readfromfile("loss_th_normal.txt", 1)
    loss_th_normal = np.log(loss_th_normal)
    iters = range(0,4000, 10)

    plt.title('Alphas')
    plt.plot(iters, alpha_adapt, label='Adaptive Attack')
    plt.plot(iters, alpha_normal, label='Normal Attack')
    plt.xlabel('Iteration', fontsize=14)
    plt.legend()
    plt.show()

    plt.title('Threshold Loss')
    plt.plot(iters, loss_th_adapt, label='Adaptive Attack')
    plt.plot(iters, loss_th_normal, label='Normal Attack')

    plt.legend()
    print(plt.gca().get_ylim(), plt.gca().get_xlim())
    plt.xlabel('Iteration', fontsize=14)
    #plt.ylabel('WER (%)', fontsize=14)
    plt.show()




main()