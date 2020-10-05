#!/bin/sh

cd ..
cd ..
cd ..

python2.7 -m pip install librosa
python2.7 -m pip install Cython
python2.7 -m pip install pyroomacoustics
python2.7 -m pip install pydub



cd cleverhans || exit

git checkout clean-testing

cd examples/adversarial_asr/model/ || exit

#wget http://cseweb.ucsd.edu/~yaq007/ckpt-00908156.data-00000-of-00001



cd ..
cd ..
cd ..
cd ..

apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        gcc-4.8 g++-4.8 gcc-4.8-base \
        git \
        less \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng-dev \
        libzmq3-dev \
        lsof \
        pkg-config \
        python \
        python-dev \
        python-tk \
        rsync \
        software-properties-common \
        sox \
        unzip \
        vim \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 100 &&\
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 100

curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

python2.7 -m pip --no-cache-dir install \
        Pillow \
        h5py \
        ipykernel \
        jupyter \
        jupyter_http_over_ws \
        matplotlib \
        numpy \
        pandas \
        recommonmark \
        scipy \
        sklearn \
        sphinx \
        sphinx_rtd_theme \
        && \
    python -m ipykernel.kernelspec

jupyter serverextension enable --py jupyter_http_over_ws

python2.7 -m pip uninstall -y tensorflow

python2.7 -m pip uninstall -y protobuf

pip uninstall -y protobuf

pip uninstall -y tensorflow

python2.7 -m pip --no-cache-dir install tensorflow-gpu==1.13.1

mkdir /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/0.17.2/bazel-0.17.2-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-0.17.2-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-0.17.2-installer-linux-x86_64.sh

mkdir lingvo_compiled

export LINGVO_DEVICE="gpu"
export LINGVO_DIR=$HOME/lingvo

cd lingvo || exit

bazel build -c opt --config=cuda //lingvo:trainer
cp -rfL bazel-bin/lingvo/trainer.runfiles/__main__/lingvo ../lingvo_compiled


cd ../lingvo_compiled/ || exit

mv lingvo ../cleverhans/examples/adversarial_asr/

cd ./cleverhans/examples/adversarial_asr/ || exit






