FROM ubuntu:18.04 AS builder
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV CMAKE_VERSION 3.20.0

RUN apt-get update && apt-get install -y apt-transport-https ca-certificates && \
    apt-get update && \
    apt-get install -y build-essential git vim sudo \
    libhdf5-dev libopenblas-dev \
    libboost-dev libboost-filesystem-dev libboost-system-dev \
    libboost-regex-dev libboost-thread-dev \
    libncurses5-dev \
    libssl-dev \
    python3.7-dev \
    python3-distutils swig \
    tzdata \
    ninja-build \
    parallel \
    curl wget \
    # for opencv
    libgl1 \
    libnuma1 libatlas-base-dev \
    unzip vim \
    graphviz \
    gdb \
    # for document
    texlive-xetex \
    && apt-get clean && \
    # config python
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 0 && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.7 get-pip.py && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 0 && \
    # install cmake
    wget https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERSION/cmake-$CMAKE_VERSION-Linux-x86_64.sh \
    --no-check-certificate -q -O /tmp/cmake-install.sh && \
    chmod u+x /tmp/cmake-install.sh && \
    /tmp/cmake-install.sh --skip-license --prefix=/usr/local && \
    rm /tmp/cmake-install.sh && \
    #rm -rf glog && \
    #git clone git@github.com:google/glog.git && \
    #cd glog && git checkout v0.5.0 && mkdir -p build && cd build && cmake ../ && make -j 4 && make install &&\
    pip3 install \
    argcomplete \
    Cython \
    decorator \
    dash \
    dash-bootstrap-components \
    dash-draggable \
    dash-cytoscape \
    dash-split-pane \
    dash-table \
    enum34 \
    gitpython \
    graphviz \
    grpcio \
    ipykernel \
    ipython \
    jedi \
    Jinja2 \
    jupyterlab \
    jsonschema==3.2.0 \
    kaleido \
    leveldb \
    lmdb \
    matplotlib \
    networkx \
    nose \
    numpy \
    opencv-contrib-python \
    opencv-python \
    opencv-python-headless \
    packaging \
    paddle2onnx \
    paddlepaddle \
    pandas \
    paramiko \
    Pillow \
    plotly \
    ply \
    # https://github.com/sophgo/docker/actions/runs/3616914101/jobs/6095296001#step:5:3166
    protobuf \
    pybind11[global] \
    pycocotools \
    python-dateutil \
    python-gflags \
    pyyaml \
    scikit-image \
    scipy \
    six \
    sphinx sphinx-autobuild sphinx_rtd_theme rst2pdf \
    termcolor \
    tqdm \
    wheel && \
    pip3 install onnx==1.13.0 onnxruntime==1.14.0 onnxsim && \
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu &&  \
    rm -rf ~/.cache/pip/*

RUN TZ=Asia/Shanghai \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && dpkg-reconfigure -f noninteractive tzdata \
    && ln -s /usr/local/bin/pip /usr/bin/pip \
    # install some fonts
    && wget "http://mirrors.ctan.org/fonts/fandol.zip" -O /usr/share/fonts/fandol.zip \
    && unzip /usr/share/fonts/fandol.zip -d /usr/share/fonts \
    && rm /usr/share/fonts/fandol.zip \
    && git config --global --add safe.directory '*' \
    # install ccache
    && wget "https://github.com/ccache/ccache/releases/download/v4.7.4/ccache-4.7.4-linux-x86_64.tar.xz" -O /tmp/ccache-4.7.4-linux-x86_64.tar.xz \
    && tar xf /tmp/ccache-4.7.4-linux-x86_64.tar.xz -C /tmp \
    && mv /tmp/ccache-4.7.4-linux-x86_64/ccache /usr/local/bin/ \
    && rm -rf /tmp/*

#COPY --from=builder /usr/local /usr/local
ENV LC_ALL=C.UTF-8

WORKDIR /workspace

