import pandas as pd
import torch
import numpy as np

from BPPN import BPPN
from Prepare_PUdata import Prepare_PUdata
from NNPU import NNPU
from LoadData import LoadData

def my_model(X_df, Y_df, args, reliable_ratio=0.2, h1=None,h2=None):

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # 特征维度
    input_dim = X_df.shape[1]

    # 划分数据(cutData隐藏部分正样本，划分训练集和测试集)
    prepare_PUdata = Prepare_PUdata(X_df, Y_df)

    # PU data indexes
    label_indexes = np.array(prepare_PUdata.get_label_indexes())
    unlabel_indexes = np.array(prepare_PUdata.get_unlabel_indexes())

    train_set = prepare_PUdata.get_train_set();
    test_set = prepare_PUdata.get_test_set();

    prior = prepare_PUdata.get_prior()

    train_loader = torch.utils.data.DataLoader(train_set,drop_last=True, batch_size=args.train_batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, drop_last=True, batch_size=args.eval_batch_size, shuffle=True, **kwargs)

    ## nnPU
    nnpu = NNPU(train_loader, input_dim, 1, prior, args)

    nnpu_model = nnpu.run_train();
    ##阈值划分

    label_num=prepare_PUdata.get_label_num();
    unlabel_set=prepare_PUdata.get_unlabel_set()
    unlabel_loader = torch.utils.data.DataLoader(unlabel_set, batch_size=args.eval_batch_size, shuffle=True, **kwargs)
    h1, h2, result = pseudo_label(unlabel_loader, label_num, nnpu_model,reliable_ratio,h1,h2)

    pn_label=np.hstack((result,np.ones(len(label_indexes))))
    indexes=np.hstack((unlabel_indexes,label_indexes))
    pn_label_df= pd.DataFrame(pn_label, index=indexes, columns={'label'})

    pn_train_set=LoadData(indexes, X_df, pn_label_df)

    pn_train_loader = torch.utils.data.DataLoader(pn_train_set,drop_last=True, batch_size=args.train_batch_size, shuffle=True,
                                                  **kwargs)

    # BPPN
    bppn = BPPN(pn_train_loader, test_loader, input_dim, 2, args)
    bppn.run_train()
    precision, recall, f1, acc = bppn.test()
    return precision, recall, f1, acc

def pseudo_label(unlabel_loader, label_num, model,reliable_ratio=None,h1=None,h2=None):
    result=None
    for data,target in unlabel_loader:
        output = model(data)
        pred = output.detach().numpy();
        if result is None:
            result=pred.flatten()
        else:
            result=np.hstack((result,pred.flatten()))

    # 无标记样本的proba
    result = 1 / (1 + np.exp(-result))
    result = result.flatten()

    #reliable_ratio为None表示按照绝对阈值选取可靠正负样本
    #不为None按照比例确定相对阈值
    if reliable_ratio:
        sort_result = result.copy()
        sort_result.sort(axis=-1, kind='quicksort', order=None)
        reliable_neg_num = int((1+reliable_ratio)*label_num)
        reliable_pos_num = int(0.2*label_num)
        h2 = sort_result[reliable_neg_num]
        result_reverse = sort_result[::-1]
        h1 = result_reverse[reliable_pos_num]
    result = np.where(result > h1, 1, result)
    result = np.where(result < h2, 0, result)
    return h1, h2, result