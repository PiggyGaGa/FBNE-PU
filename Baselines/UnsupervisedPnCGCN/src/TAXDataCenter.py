import sys
import os
import pandas as pd
from collections import defaultdict
import numpy as np
import time
class TAXDataCenter(object):
	"""docstring for DataCenter"""
	def __init__(self, basic_feature_file, edges_file, label_file):
		super(TAXDataCenter, self).__init__()
		self.basic_feature_file = basic_feature_file
		self.edges_file = edges_file
		self.label_file = label_file
		
	def load_dataSet(self, dataSet):
		begin_time = time.time()
		print('开始加载 {} 的数据集'.format(dataSet))
		feat_data = []
		labels = [] # label sequence of node
		node_map = {} # map node name to Node_ID, 这个会在后面用到
		label_map = {} # map label to Label_ID

		Basic_feature = pd.read_pickle(self.basic_feature_file)
		node_names = Basic_feature.index.values.tolist()
		## 基本特征
		feat_data = []
		i = 0
		for nsrdzdah, feat in Basic_feature.iterrows():
			feat_data.append(feat.values)
			node_map[nsrdzdah] = i
			i += 1
		feat_data = np.asarray(feat_data)

		

		# 邻接
		adj_lists = defaultdict(set)
		Edges = pd.read_pickle(self.edges_file)
		for edge in Edges.to_numpy():
			assert edge.shape[0] == 2
			xfnsrdzdah = node_map[edge[0]]
			gfnsrdzdah = node_map[edge[1]]
			adj_lists[xfnsrdzdah].add(gfnsrdzdah)
			adj_lists[gfnsrdzdah].add(xfnsrdzdah)
		
		


		## 标签
		Label_pd = pd.read_pickle(self.label_file)
		label_map = {}  ## label 存储的是纳税人对应的label
		# import pdb
		# pdb.set_trace()
		for nsrdzdah, single_label in zip(Label_pd.index, Label_pd.values.flatten()):
			if not single_label in label_map:
				label_map[nsrdzdah] = single_label
			labels.append(single_label)
		labels = np.asarray(labels, dtype=np.int64)


		# import pdb
		# pdb.set_trace()
		assert len(feat_data) == len(labels) == len(adj_lists)

		
		# train test valid index 分割
		test_indexs, val_indexs, train_indexs = self._split_data(labels)
		
		setattr(self, dataSet+'_test', test_indexs)
		setattr(self, dataSet+'_val', val_indexs)
		setattr(self, dataSet+'_train', train_indexs)
		setattr(self, dataSet+'_feats', feat_data)
		setattr(self, dataSet+'_labels', labels)
		setattr(self, dataSet+'_adj_lists', adj_lists)
		setattr(self, dataSet+'_node_map', node_map)
		setattr(self, dataSet+'_node_names', node_names)

		end_time = time.time()
		print('完成加载 {} 的数据集'.format(dataSet))
		print('加载数据集所用时间为：{:.3f} s'.format(end_time-begin_time))


	def _split_data(self, labels, test_split = 3, val_split = 6):

		positive_idx = np.where(labels==1)[0]
		negative_idx = np.where(labels==0)[0]

		positive_len = len(positive_idx)
		negative_len = len(negative_idx)

		# positive valid, test, train indices
		positive_valid_idx = np.random.choice(positive_idx, positive_len // test_split)
		positive_test_idx = np.random.choice(list(set(positive_idx) - set(positive_valid_idx)), positive_len // val_split)
		positive_train_idx = list(set(positive_idx) - (set(positive_valid_idx) | set(positive_test_idx)))


		negative_valid_idx = np.random.choice(negative_idx, negative_len // 3)
		negative_test_idx = np.random.choice(list(set(negative_idx) - set(negative_valid_idx)), negative_len // 6)
		negative_train_idx = set(negative_idx) - (set(negative_valid_idx) | set(negative_test_idx))

		test_indexs = list(set(positive_test_idx) | set(negative_test_idx))
		val_indexs = list(set(positive_valid_idx) | set(negative_valid_idx))
		train_indexs = list(set(positive_train_idx) | set(negative_train_idx))

		return test_indexs, val_indexs, train_indexs
