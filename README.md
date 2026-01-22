# Influence Shaping

## To Install

```
git clone --recurse-submodules git@github.com:AADILab/influence-shaping.git
```

### Conda Setup
Install miniconda according to their [website](https://docs.anaconda.com/miniconda/).

For Linux x86_64 operating system:
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

For Arch Linux ARM operating system:
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

For MacOS:
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -O ~/miniconda3/miniconda.sh
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

### Install Influence Library

Install the influence library and its remaining requirements
```
cd influence-shaping
pip install -e .
```

### Run Example
Now you should be able to run the example.
```
python tools/run/config.py example/results/config.yaml
```

If you are running on MacOS and you see a warning about finding
```
/Applications/Xcode_12.4.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
```

then you can use a softlink to fix this path
```
sudo ln -s /Applications/Xcode.app /Applications/Xcode_12.4.app
```

## To run tests

In the `test` folder, there are unit tests written in python that test different parts of the library function as expected. Run them with the following command.

```
python -m unittest discover -s test
```
