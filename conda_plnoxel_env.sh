# run this shell after you activate virtual env
#conda create -n plenoxel python=3.10
#conda activate plenoxel
conda install numpy
pip install imageio imageio-ffmpeg ipdb lpips
pip install pymcubes moviepy matplotlib
pip install opencv-python Pillow pyyaml tensorboard scipy
pip install -e . --verbose

