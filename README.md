# Influence Shaping

## To Install

```
git clone --recurse-submodules git@github.com:AADILab/influence-shaping.git
```

Install miniconda according to their [website](https://docs.anaconda.com/miniconda/).
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

Source the miniconda install.
```
source ~/miniconda3/bin/activate
```

Create the env for this repo.
```
conda create -n influence
conda activate influence
conda install conda-forge::cppyy=2.2.0
```

Install the remaining requirements.
```
cd influence-shaping
pip install -r requirements.txt
```

Now you should be able to run the example.
```
python prototyping/run_cli.py prototyping/configs/default.yaml
```
