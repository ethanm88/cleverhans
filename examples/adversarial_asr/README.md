# Defense for Adaptive Imperceptible Audio Adversarial Examples Using Proportional Additive Gaussian Noise

This a defense for the imperceptible audio adversarial examples from the ICML 2019 paper ["Imperceptible, Robust and Targeted Adversarial Examples for Automatic Speech Recognition"](http://proceedings.mlr.press/v97/qin19a.html). The details of all the models implemented here can be found in the [paper](http://proceedings.mlr.press/v97/qin19a.html).

## Dependencies
*   Python 2.7
*   a TensorFlow [installation](https://www.tensorflow.org/install/) (Tensorflow 1.13 is supported for this version of Lingvo system),
*   a `C++` compiler (only g++ 4.8 is officially supported),
*   the bazel build system,
*   librosa (```pip install librosa```),
*   Cython (```pip install Cython```),
*   pyroomacoustics (```pip install pyroomacoustics```).

See Setup and Build Model section for quick installation of all dependencies.

## Lingvo ASR system

The automatic speech recognition (ASR) system used in this paper is [Lingvo system](https://github.com/tensorflow/lingvo). To run this code, you need to first download the forked version [here](https://github.com/ethanm88/lingvo).

## Setup and Build Model

In order to build the Lingvo Model and install all dependencies run the setup script in this (adversarial_asr) directory: 
```sh setup.sh```.

## Imperceptible Adversarial Examples

Currently, all the python scripts are tested on one GPU. You can use ```CUDA_VISIBLE_DEVICES=GPU_INDEX``` to choose which gpu to run the python scripts.

To generate imperceptible adversarial examples, run

```bash
python generate_imperceptible_adv.py
```

The adversarial examples saved with the name ended with "stage1" is the adversarial examples in [Carlini's work](https://arxiv.org/abs/1801.01944). Adversarial examples ended with the name "stage2" is our imperceptible adversarial examples using frequency masking threshold.

To test the accuracy of our imperceptible adversarial examples, simply run:

```bash
python test_imperceptible_adv.py --stage=stage2 --adv=True
```
You can set ```--stage=stage1``` to test the accuracy of Carlini's adversarial examples. If you set ```--adv=False```, then you can test the performance for clean examples with its corresponding original transcriptions.

## Robust Adversarial Examples
To generate robust adversarial examples that are simulated playing over-the-air in the simulated random rooms, we need to first generate the simulated room reverberations.
```bash
python room_simulator.py
```
Then you can run the following command to generate robust adversarial examples.
```
python generate_robust_adv.py --initial_bound=2000 --num_iter_stage1=2000
```
In the paper, we test the last 100 audios in the ```./util/read_data_full.txt``` and we set the parameter ```initial bound``` and ```num_iter_stage1``` as ```2000``` in our experiments.

Empirically, for longer audios, you might need to increase the ```initial bound``` of perturbation to generate robust adversarial examples that can successfully attack the simulated rooms. Correspndingly, you also need to increase ```num_iter_stage1``` to allow the adversarial generation to converge. You can tune the training parameters in ```generate_robust_adv.py``` to play with your data.

To test the performance of robust adversarial examples, simply run 
```
python test_robust_adv.py --stage=stage2 --adv=True
```
If you want to test the performance of clean examples played in the simulated rooms, you can set ```--adv=False```.

## Defense Evaluation

Once you have adversarial examples, you can evaluate their effectiveness against a defense (you must be in the ```adversarial_asr``` directory). If you plan to run our defense, first you must construct defensive perturbations.  
```
mkdir noisy_data --num_iter=100 --factor=0.00
```

You can change the number of defensive perturbations created by changing  ```--num_iter```. Note that defensive perturbations will be randomly pulled from the batch during each evaluation. Next, run the evaluation.
```
python2.7 evaluate_defense_mod.py --type_defense=0
```
```--type_defense``` can be set to 0 for our defense, 1 for an MP3 compression defense, and 2 for a Quantization defense (Song et al.). Currently, you can also set the ```--perturb_location``` flag to the name of the perturbation wav file.

## Adaptive Adversarial Examples


## Citation
If you find the code or the models implemented here are useful, please cite this paper:

```
@InProceedings{pmlr-v97-qin19a,
  title = 	 {Imperceptible, Robust, and Targeted Adversarial Examples for Automatic Speech Recognition},
  author = 	 {Qin, Yao and Carlini, Nicholas and Cottrell, Garrison and Goodfellow, Ian and Raffel, Colin},
  booktitle = 	 {Proceedings of the 36th International Conference on Machine Learning},
  pages = 	 {5231--5240},
  year = 	 {2019},
  volume = 	 {97},
  series = 	 {Proceedings of Machine Learning Research},
  publisher = 	 {PMLR},
}
```

## Acknowledgement
This code is based on Lingvo ASR system. Thanks to the contributors of the Lingvo.

```
@article{shen2019lingvo,
  title={Lingvo: a modular and scalable framework for sequence-to-sequence modeling},
  author={Shen, Jonathan and Nguyen, Patrick and Wu, Yonghui and Chen, Zhifeng and Chen, Mia X and Jia, Ye and Kannan, Anjuli and Sainath, Tara and Cao, Yuan and Chiu, Chung-Cheng and others},
  journal={arXiv preprint arXiv:1902.08295},
  year={2019}
}
```

