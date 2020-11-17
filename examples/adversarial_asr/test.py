'''
Original File:
- Generates Imperceptible Targeted Adversarial Examples according to appraoch developed by Qin et al.
- Serves as a template - unmodified from previous source code
- copy code from master

'''
import tensorflow as tf

from absl import flags
from absl import app


def main(argv):
    print("hello1");
    if tf.test.gpu_device_name():
        print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")


if __name__ == '__main__':
    app.run(main)