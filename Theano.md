

First of all, you should update your system:

```
sudo apt-get update
sudo apt-get upgrade
```



### Theano

Prerequisites (incl. Python):

```
sudo apt-get install python-dev python-pip libblas-dev liblapack-dev cmake
sudo pip install numpy, scipy, cython
```
Install gpuarray:

http://deeplearning.net/software/libgpuarray/installation.html

```
git clone https://github.com/Theano/libgpuarray.git
cd libgpuarray
mkdir Build
cd Build
cmake .. -DCMAKE_BUILD_TYPE=Release # or Debug if you are investigating a crash
make
sudo make install
cd ..

sudo python setup.py build
sudo python setup.py install

sudo ldconfig
```

Install Theano:

```
git clone git://github.com/Theano/Theano.git
cd Theano
sudo pip install -e .
```

### .theanorc

Adjust .theanorc in your home directory to select a GPU and fix random seeds:

```
[global]
floatX = float32
device = cuda0

[lib]
cnmem = 1

[nvcc]
flags=-D_FORCE_INLINES
fastmath = True

[blas]
ldflags = -lopenblas

[cuda]
root = /usr/local/cuda-9.0/bin

[dnn]
include_path = /usr/lib/cuda/include
library_path = /usr/lib/cuda/lib64
```

### Lasagne

Clone the repository and install Lasagne:

```
sudo pip install https://github.com/Lasagne/Lasagne/archive/master.zip
```

### OpenCV

We use OpenCV for image processing; you can install the cv2 package for Python running this command:

```
sudo pip install python-opencv
```

### Libav

The audio processing library Librosa uses the Libav tools:

```
sudo apt-get install libav-tools
