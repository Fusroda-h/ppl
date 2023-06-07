set -e

apt-get update
apt-get upgrade -y
apt-get install cmake gcc g++ vim -y

# Build eigen
if [ -d "eigen-3.4.0/" ]; then
    rm -r eigen-3.4.0/
fi
tar -xf eigen-3.4.0.tar.gz
cd eigen-3.4.0
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=// ..
make install

# pip package install
pip install pybind11 numpy opencv-python tqdm matplotlib scikit-learn scikit-image torch pandas
apt-get install libgl1-mesa-glx -y

# Build poselib
cd ../../PoseLib
if [ -d "_buld" ]; then
    rm -r _build
fi
mkdir _build
cmake -S . -B _build/ -DPYTHON_PACKAGE=ON -DCMAKE_INSTALL_PREFIX=_install
cmake --build _build/ --target install -j 8
cmake --build _build/ --target pip-package
cmake --build _build/ --target install-pip-package

# install colmap
cd ../../
apt-get install -y git cmake build-essential libboost-program-options-dev libboost-filesystem-dev libboost-graph-dev libboost-system-dev libboost-test-dev libeigen3-dev libsuitesparse-dev libfreeimage-dev libmetis-dev libgoogle-glog-dev libgflags-dev libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev
# Ceres solver
git clone https://ceres-solver.googlesource.com/ceres-solver-2.1.0
mkdir ceres-bin && cd ceres-bin
cmake ../ceres-solver-2.1.0
make -j3
make test
make install
# Colmap
cd ../
git clone https://github.com/colmap/colmap.git
cd colmap
git checkout dev
mkdir build
cd build
cmake ..
make -j
make install
