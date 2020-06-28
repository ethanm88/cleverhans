import matplotlib.pyplot as plt

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

    iters = range(0,4000, 10)

    plt.plot(iters, alpha_adapt, label='Alpha')
    plt.xlabel('Iteration', fontsize=14)
    plt.legend()
    plt.show()
    plt.plot(iters, loss_th_adapt, label='Loss_Th')

    plt.legend()
    print(plt.gca().get_ylim(), plt.gca().get_xlim())
    plt.xlabel('Iteration', fontsize=14)
    #plt.ylabel('WER (%)', fontsize=14)
    plt.show()




main()