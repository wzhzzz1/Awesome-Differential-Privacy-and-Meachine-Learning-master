conda create --name pdldpfl python=3.8
conda activate pdldpfl
conda install -c anaconda pip
conda install -c anaconda setuptools wheel
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cudatoolkit=11.2 -c pytorch
pip install -r ./requirements.txt
