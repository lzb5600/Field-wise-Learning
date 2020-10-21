# Field-wise Learning for Multi-field Categorical Data

This repository is the official implementation of [Field-wise Learning for Multi-field Categorical Data](https://arxiv.org/abs/2030.12345). 

## Requirements
The code has been tested with:
- Python 3.6.8
- PyTorch 1.1.0
- lmdb 0.96
- tqdm 4.32.1



## Training and Evaluation
1. Download the [Avazu](https://www.kaggle.com/c/avazu-ctr-prediction) and [Criteo](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/) datasets.

2. To train and evaluate the model(s), run following command (see full input arguments via python run_fwl.py --help):
	```run
	python run_fwl.py --dataset-path <path_to_data>
	```
For example, to train the model on Criteo datasets, run:
```run
python run_fwl.py  --dataset-path ./data/criteo/train.csv --ebd-dim 1.6 --log-ebd --lr 0.01 --wdcy 1e-6 --include-linear --reg-lr 1e-3 --reg-mean --reg-adagrad
```
## Citation
If you find this repository helpful, please consider to cite the following paper:
```

```
