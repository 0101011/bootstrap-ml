!pip install tf-nightly-gpu-2.0-preview

!wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda-repo-ubuntu1604-10-0-local-10.0.130-410.48_1.0-1_amd64 -O cuda-repo-ubuntu1604-10-0-local-10.0.130-410.48_1.0-1_amd64.deb
!dpkg -i cuda-repo-ubuntu1604-10-0-local-10.0.130-410.48_1.0-1_amd64.deb
!apt-key add /var/cuda-repo-10-0-local-10.0.130-410.48/7fa2af80.pub
!apt-get update
!apt-get install cuda
!pip install tf-nightly-gpu-2.0-preview

import tensorflow as tf

print(tf.__version__)
