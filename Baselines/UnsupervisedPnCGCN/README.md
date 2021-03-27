


## unsupervised PnCGCN 

This package implement the unsupervised PnCGCN, and this package is on the basis of [graphSAGE](https://github.com/twjiang/graphSAGE-pytorch) pytorch.
We changed the sampling operation according to the idea of PnCGCN


## Author
name: PiggyGaGa
email: liana_gyd@163.com



## Environment settings

- python==3.6.8
- pytorch==1.7.0
- pandas==1.1.4
- scikit-learn==0.23.2



## Basic Usage

**Main Parameters:**

```
--dataSet     The input graph dataset. (default: TaxS)
--agg_func    The aggregate function. (default: Mean aggregater)
--epochs      Number of epochs. (default: 50)
--b_sz        Batch size. (default: 20)
--seed        Random seed. (default: 824)
--unsup_loss  The loss function for unsupervised learning. ('margin' or 'normal', default: normal)
--config      Config file. (default: ./src/TAXConfig.conf)
--cuda        Use GPU if declared.
```

**Learning Method**

The user can specify a learning method by --learn_method, 'sup' is for supervised learning, 'unsup' is for unsupervised learning, and 'plus_unsup' is for jointly learning the loss of supervised and unsupervised method.

**Example Usage**

To run the unsupervised model on Cuda:
```
python -m main --epochs 100 --learn_method unsup --cuda --gcn --b_sz 256 --dataSet TaxH --config src/experiments.conf```

