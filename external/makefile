
nvcc = /usr/local/cuda/bin/nvcc
cudalib = /usr/local/cuda/lib64
tensorflow = /usr/local/lib/python3.6/dist-packages/tensorflow/include #/home/wnylol/tools/anaconda2/envs/tensorflow/lib/python2.7/site-packages/tensorflow/include

TF_CFLAGS=`python -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))"`
TF_LFLAGS=`python -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()))"`


ALL = tf_approxmatch_so.so tf_approxmatch_g.cu.o tf_nndistance_so.so tf_nndistance_g.cu.o

all: $(ALL)

tf_approxmatch_so.so: tf_approxmatch_g.cu.o tf_approxmatch.cpp
	g++ -std=c++11 -shared tf_approxmatch_g.cu.o tf_approxmatch.cpp -o tf_approxmatch_so.so -fPIC $(TF_CFLAGS) $(TF_LFLAGS) -L$(cudalib) -O2

tf_approxmatch_g.cu.o: tf_approxmatch_g.cu
	$(nvcc) -std=c++11 -c tf_approxmatch_g.cu -o tf_approxmatch_g.cu.o -I $(TF_CFLAGS) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 -Xcudafe="--diag_suppress=2886"

tf_nndistance_so.so: tf_nndistance_g.cu.o tf_nndistance.cpp
	g++ -std=c++11 -shared tf_nndistance_g.cu.o tf_nndistance.cpp -o tf_nndistance_so.so -fPIC $(TF_CFLAGS) $(TF_LFLAGS) -L$(cudalib) -O2

tf_nndistance_g.cu.o: tf_nndistance_g.cu
	$(nvcc) -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu $(TF_CFLAGS) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 -Xcudafe="--diag_suppress=2886"

clean:
	rm -f $(ALL)

.PHONY: clean
