## CasTGAN

This is the source code for the article *CasTGAN: Cascaded Generative Adversarial Network for Realistic Tabular Data Synthesis*. Please cite this work accordingly.

#### System Requirements

This code was developed in Python 3.10. The repository can simply be cloned to a local directory. To install the libraries, run either `python3.10 -m pip install -r requirements_gpu.txt` if on a GPU is installed, otherwise running `python3.10 -m pip install -r requirements_cpu.txt` should also work on CPU based machines.

#### Datasets

The datasets used in this work are publicly available and are referenced in the article. Due to the limited upload size on Github, the bank dataset is uploaded and can be found under the Data folder. Generated synthetic data can be found under the Generated_Data folder. If no perturbations to the auxiliary learners were introduced, the synthetic data will be saved under the wb_0 subfolder. 

#### Code Demo

CasTGAN can be executed by writing the following line in CLI:

```
python3.10 -m main --dataset="bank" --epochs=300
```

"bank" can be replaced by any dataset given that the dataset with a "_train.csv" suffix and located under the Data directory. A supporting "_dtypes.pkl" file is optional but recommended to ensure that the correct datatypes are read. The unpickled file is essentially a dictionary with the column names as keys and the data types as values. 