# install pytorch 
if [ "$1" = "cpu" ]; then
    conda install pytorch torchvision torchaudio cpuonly -c pytorch
else
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
fi

# install numpy and matplotlib
conda install numpy matplotlib -y

# install opencv
conda install -c conda-forge opencv -y
