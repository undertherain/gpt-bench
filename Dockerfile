from ubuntu:22.04

WORKDIR /workdir/

RUN apt-get update && \
    apt-get install -yqq --no-install-recommends software-properties-common && \
    apt-get install -yqq sudo && \
    apt-get install -yqq git && \
    apt-get install -yqq cmake && \
    apt-get install -yqq pkg-config && \
    apt-get install -yqq zip && \
    apt-get install -yqq zlib1g-dev && \
    apt-get install -yqq unzip && \
    apt-get install -yqq wget && \
    apt-get install -yqq libopenblas-dev && \
    apt-get install -yqq libhdf5-dev ninja-build libuv1-dev && \
    apt-get clean

#    apt-get install -yqq build-essential && \
RUN apt-get install -y gcc-10 g++-10 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 1 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 1

#ENV CC=/usr/bin/gcc-10
#ENV CXX=/usr/bin/g++-10

RUN apt-get -yqq install --no-install-recommends python3-dev  python3-pip  python3-wheel  python3-setuptools python3-yaml python3-numpy && \
    apt-get clean


#RUN apt-get -yqq install --no-install-recommend intel-mkl libmkl-dev && \
#    apt-get clean

RUN apt-get install -yqq nvidia-cuda-toolkit && \
    apt-get clean

RUN wget https://developer.download.nvidia.com/compute/redist/cudnn/v8.2.4/cudnn-11.4-linux-x64-v8.2.4.15.tgz && \
    tar -xvf cudnn-11.4-linux-x64-v8.2.4.15.tgz && \
    cp -P ./cuda/lib64/libcudnn* /usr/lib/x86_64-linux-gnu/ && \
    cp ./cuda/include/cudnn*.h /usr/include/x86_64-linux-gnu/ && \
    rm -rf ./cuda && \
    rm cudnn-11.4-linux-x64-v8.2.4.15.tgz

#/tmp/nvidia-cudnn/cuda/lib64/  /usr/lib/x86_64-linux-gnu/
#/tmp/nvidia-cudnn/cuda/include/ /usr/include/x86_64-linux-gnu/

#RUN echo "cudnn cudnn/license_preseed select ACCEPT" | debconf-set-selections
#RUN apt-get install nvidia-cudnn
# && 
#    apt-get clean

RUN pip3 install typing-extensions jinja2 filelock fsspec networkx sympy jupyter



ENV USE_CUDNN=1
ENV BUILD_TEST=0
ENV _GLIBCXX_USE_CXX11_ABI=1


RUN git clone --recursive https://github.com/pytorch/pytorch && \
    cd pytorch && \
    python3 setup.py install && \
    cd .. && \
    rm -rf pytorch


RUN pip3 install deepspeed
RUN pip3 install --upgrade pip

RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./ && \
    cd .. && \    
    rm -rf apex


# RUN pip3 install six numpy wheel setuptools mock
# RUN pip3 install pydataset
# RUN pip3 install py-cpuinfo
# RUN pip3 install system_query[cpu,hdd,ram,swap]
# RUN git clone https://github.com/undertherain/benchmarker.git

#ENV USE_FBGEMM=0
#ENV USE_NNPACK=0
#ENV USE_QNNPACK=0
#ENV USE_DISTRIBUTED=0
#ENV USE_SYSTEM_NCCL=0
#ENV BUILD_CAFFE2_OPS=1
# ENV MAX_JOBS=36
#ENV USE_NCCL=ON
#ENV USE_CUDA=ON
#ENV BLAS=OpenBLAS
#ENV USE_MKLDNN=ON
#ENV USE_MKLDNN_CBLAS=ON
#ENV MKLDNN_USE_NATIVE_ARCH=ON
#ENV CC=gcc
#ENV CXX=g++
#ENV ARCH_OPT_FLAGS="-msse4.2 -msse2avx -gno-inline-points -fearly-inlining -march=native -O3"
#ENV CMAKE_C_FLAGS="-msse4.2 -msse2avx -gno-inline-points -fearly-inlining -march=native -O3"
#ENV CMAKE_CXX_FLAGS="$CMAKE_C_FLAGS"
#ENV CFLAGS="$CMAKE_C_FLAGS"

#RUN git clone --recursive https://github.com/pytorch/pytorch.git
#RUN pip3 install ninja pyyaml cffi typing

#WORKDIR /workdir/pytorch
#RUN git checkout v1.4.0 && git submodule update --init --recursive 
#RUN python3 setup.py install
#WORKDIR /workdir/
#RUN git clone https://github.com/dl4fugaku/dl4fugaku.git
#RUN apt-get install -y linux-tools-generic
#RUN rm /usr/bin/perf
#RUN ln -s /usr/lib/linux-tools/*/perf /usr/bin/perf
