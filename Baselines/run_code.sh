#!/bin/bash -l

conda activate python35
python main_TaxS.py
python main_TaxZ.py
python main_TaxH.py
python main_gcn_pncgcn_TaxH.py
python main_gcn_pncgcn_TaxS.py
python main_gcn_pncgcn_TaxZ.py