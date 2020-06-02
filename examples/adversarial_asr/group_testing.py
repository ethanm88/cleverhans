from absl import app

import evaluate_defense_mod
import apply_defense_mod
import matplotlib as plt

def graph(all_stan_dev, all_wer_adv, all_wer_benign):

    plt.plot(all_stan_dev, all_wer_adv, label="Adversarial WER")
    plt.plot(all_stan_dev, all_wer_benign, label="Benign WER")

    plt.legend()
    plt.xlabel('Sigma Constant', fontsize=14)
    plt.ylabel('Word Error Rate (%)', fontsize=14)

    plt.show()

def main(argv):
    initial = 0
    increment = 0.05
    maximum = 1.01
    all_wer_benign = []
    all_wer_adv = []
    all_stan_dev = []

    for sd in range(initial, maximum, increment):
        all_stan_dev.append(sd)
        # apply defense:
        apply_defense_mod.save_audios(sd)

        # find word error rates:
        temp_adv, temp_benign = evaluate_defense_mod.get_WERs()
        all_wer_adv.append(temp_adv)
        all_wer_benign.append(temp_benign)

    print("WER Adversarial", all_wer_adv)
    print("WER Benign", all_wer_benign)







if __name__ == '__main__':
    app.run(main)