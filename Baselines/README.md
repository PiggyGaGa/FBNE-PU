## FBNE-PU   Baselines

```
file: readme.md
author: PiggyGaGa
email: liana_gyd@163.com
```


## run example
```
conda activate env-name   
python main_TaxS.py   # most of baselines for TaxS
python main_TaxZ.py  # most of baselines for TaxZ
python main_TaxH.py   # most of baselines for TaxH
python main_gcn_pncgcn_TaxH.py   # gcn, pncgcn for TaxH
python main_gcn_pncgcn_TaxS.py   # gcn, pncgcn for TaxS
python main_gcn_pncgcn_TaxZ.py   # gcn, pncgcn for Taxz

For Unsupervised PnCGCN Please refer to `UnsupervisedPnCGCN` folder
```

## algorithms

| algorithms             | basicfeature   |
|----------------|--------|
| DeepWalk       | No     |
| Node2vec       | No     |
| Walklets       | No     |
| SDNE           | No     |
| NetRA          | No     |
| GraphAttention | No     |
| GraRep         | No     |
| DNGR           | No     |
| GCN            | yes |
| PnCGCN      | yes |
| unsupervised-PnCGCN      | yes |


## Data
Our data is tax related data, which belongs to sensitive data and needs to be kept strictly confidential. Therefore, we cannot disclose the information related to the data. If you want to cooperate, please do not hesitate to contact us