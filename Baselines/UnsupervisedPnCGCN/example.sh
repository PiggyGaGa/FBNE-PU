#!/bin/bash

## TaxZ datset   
job_cmd='python -m main --epochs 1 --learn_method unsup --cuda 1 --gcn --b_sz 300 --dataSet TaxZ'

## cora
job_cmd='python -m main --epochs 1 --learn_method unsup --cuda --gcn --b_sz 256 --dataSet TaxH --config src/experiments.conf'

eval $job_cmd
