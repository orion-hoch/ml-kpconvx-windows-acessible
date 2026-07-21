# KPConvX: Standalone code (keops-free / Windows-compatible fork)

This folder contains a standalone version of KPConvX code.

**Fork note:** this fork replaces the PyKeOps neighbor-search backend with
[torch-cluster](https://github.com/rusty1s/pytorch_cluster).
PyKeOps JIT-compiles CUDA kernels at runtime and only supports Linux/macOS;
torch-cluster ships prebuilt CUDA and CPU kernels for Linux, Windows, and macOS,
so KPConvX now runs on native Windows and no longer needs `nvcc` or a host C++
compiler at runtime anywhere. Semantics are unchanged (exact KNN, squared
distances, row-sorted, shadow indices). There is `Standalone/KPConvX/tests/test_neighbors.py`
for the parity tests against a brute-force reference. The `Pointcept-wrapper/`
tree carries the same torch-cluster port.

## Setup

Our code was tested with multiple environments and should be straightforward to setup. Adapt the following lines to your environment and version of CUDA:

```bash
conda create -n kpconvx python=3.10
conda activate kpconvx
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install easydict h5py matplotlib numpy scikit-learn timm
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install 'pyvista[all,trame]' jupyterlab
```

The same steps work on Windows (use the matching torch-cluster wheel index for
your torch/CUDA build; CPU-only wheels exist too).

## Prepare data

### S3DIS

We use the preprocessed data from Pointcept that you can download [here](https://huggingface.co/datasets/Pointcept/s3dis-compressed). Please agree with the official license before downloading it. From the `s3dis.tar.gz` archive, extract the `s3dis` folder to the `Standalone/data` directory.

### ScanObjectNN

We use the h5_files from [official website](https://hkust-vgd.github.io/scanobjectnn/). Download the preprocessed version and place it in the `Standalone/data` directory. Rename the folder `h5_files` to `ScanObjectNN`. You should have (and only need) the three following files:

```
data/ScanObjectNN/main_split/test_objectdataset_augmentedrot_scale75_1024_fps.pkl
data/ScanObjectNN/main_split/test_objectdataset_augmentedrot_scale75.h5
data/ScanObjectNN/main_split/training_objectdataset_augmentedrot_scale75.h5
```

## Experiments

### Training networks

We provide scripts to train our model on ScanObjectNN or S3DIS.

```bash
# Training on ScanObjectNN
./train_ScanObjectNN.sh

# Training on S3DIS
./train_S3DIS.sh
```

Follow instructions in these scripts to change parameters.

### Plotting functions

We provide plotting functions to plot the performances during and after training. They are in the `plot_ScanObj.py` and `plot_S3DIS.py` script. Here are

- Step 1: Define the dates of the logs you want to plot in the experiment functions. See example functions `experiment_name_1()` and `experiment_name_2()`

```python
def experiment_name_1():
    ...
    start = 'Log_2020-04-22_11-52-58'
    end = 'Log_2023-07-29_12-40-27'
    ...
```

- Step 2: Choose the log to show

```python
# Choose the logs to show
logs, logs_names = experiment_name_1()
```

- Step 3: Run the script

```bash
python3 experiments/ScanObjectNN/plot_ScanObj.py
# or
python3 experiments/S3DIS/plot_S3DIS.py
```

Once the trainings are finished, you can change to test mode with `perform_test = True`. This will start tests for the selected trained weights and show a summary of the test results.

### Test models

You can also directly test a trained network with the following scripts.

```bash
# Test a model on ScanObjectNN
./test_ScanObjectNN.sh

# Test a model on S3DIS
./test_S3DIS.sh
```

More detailed instructions are in these scripts.

### Pretrained weights

We provide the following pretrained models:

|   Model   |  Benchmark   |  OA   | mAcc  |  Size  |                                     Archive                                     |
| :-------: | :----------: | :---: | :---: | :----: | :-----------------------------------------------------------------------------: |
| KPConvD-L | ScanObjectNN | 89.7% | 88.5% | 80 MB  | [link](https://ml-site.cdn-apple.com/models/kpconvx/ScanObjectNN_KPConvD-L.zip) |
| KPConvX-L | ScanObjectNN | 89.1% | 87.6% | 138 MB | [link](https://ml-site.cdn-apple.com/models/kpconvx/ScanObjectNN_KPConvX-L.zip) |

|   Model   |   Benchmark   | Val mIoU |  Size  |                                 Archive                                  |
| :-------: | :-----------: | :------: | :----: | :----------------------------------------------------------------------: |
| KPConvD-L | S3DIS (Area5) |  72.3%   | 151 MB | [link](https://ml-site.cdn-apple.com/models/kpconvx/S3DIS_KPConvD-L.zip) |
| KPConvX-L | S3DIS (Area5) |  73.5%   | 169 MB | [link](https://ml-site.cdn-apple.com/models/kpconvx/S3DIS_KPConvX-L.zip) |

You can download and extract them to the result folder and use our scripts to test them.
