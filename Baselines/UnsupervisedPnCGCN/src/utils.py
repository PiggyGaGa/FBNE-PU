import sys
import os
import torch
import random
import math
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
import torch.nn as nn
import numpy as np


def validate(dataCenter, ds, graphSage, classification):
	with torch.no_grad():
		val_nodes = getattr(dataCenter, ds+'_val')
		labels = getattr(dataCenter, ds+'_labels')
		embs = graphSage(val_nodes)
		logists = classification(embs)
		_, predicts = torch.max(logists, 1)
		labels_val = labels[val_nodes]
		assert len(labels_val) == len(predicts)
		# comps = zip(labels_val, predicts.data)

		vali_f1 = f1_score(labels_val, predicts.cpu().data, average="macro")
		acc = accuracy_score(labels_val, predicts.cpu().data)
		precision, recall = precision_score(labels_val, predicts.cpu().data, average='macro'), recall_score(labels_val, predicts.cpu().data, average='macro')
		metric = {'f1': vali_f1, 'acc': acc, 'P': precision, 'R': recall}
	return metric
	# print("Validation F1:", vali_f1)

def test(dataCenter, ds, graphSage, classification, checkpoint_name = './models/graphSage_classification_best_f1.pth'):
	pth = torch.load(checkpoint_name)
	graphSage.load_state_dict(pth['graphSage'])
	classification.load_state_dict(pth['classification'])
	with torch.no_grad():
		test_nodes = getattr(dataCenter, ds+'_test')
		labels = getattr(dataCenter, ds+'_labels')
		embs = graphSage(test_nodes)
		logists = classification(embs)
		_, predicts = torch.max(logists, 1)
		labels_val = labels[test_nodes]
		assert len(labels_val) == len(predicts)
		# comps = zip(labels_val, predicts.data)
		vali_f1 = f1_score(labels_val, predicts.cpu().data, average="macro")
			
		acc = accuracy_score(labels_val, predicts.cpu().data)
		precision, recall = precision_score(labels_val, predicts.cpu().data, average='macro'), recall_score(labels_val, predicts.cpu().data, average='macro')
		# import pdb
		# pdb.set_trace()
		if ds in ['Shanghai', 'Zhejiang', 'Hebei']:
			auc = roc_auc_score(labels_val, logists.cpu().data[:, 1], multi_class='ovr')
		else:
			auc = roc_auc_score(labels_val, logists.cpu().data, multi_class='ovr')

		metric = {'f1': vali_f1, 'acc': acc, 'P': precision, 'R': recall, 'auc': auc}
	return metric

def evaluate(dataCenter, ds, graphSage, classification, device, max_vali_f1, name, cur_epoch):
	test_nodes = getattr(dataCenter, ds+'_test')
	val_nodes = getattr(dataCenter, ds+'_val')
	labels = getattr(dataCenter, ds+'_labels')

	models = [graphSage, classification]

	params = []
	for model in models:
		for param in model.parameters():
			if param.requires_grad:
				param.requires_grad = False
				params.append(param)

	embs = graphSage(val_nodes)
	logists = classification(embs)
	_, predicts = torch.max(logists, 1)
	labels_val = labels[val_nodes]
	assert len(labels_val) == len(predicts)
	comps = zip(labels_val, predicts.data)

	vali_f1 = f1_score(labels_val, predicts.cpu().data, average="micro")
	print("Validation F1:", vali_f1)

	if vali_f1 > max_vali_f1:
		max_vali_f1 = vali_f1
		embs = graphSage(test_nodes)
		logists = classification(embs)
		_, predicts = torch.max(logists, 1)
		labels_test = labels[test_nodes]
		assert len(labels_test) == len(predicts)
		comps = zip(labels_test, predicts.data)

		test_f1 = f1_score(labels_test, predicts.cpu().data, average="micro")
		print("Test F1:", test_f1)

		for param in params:
			param.requires_grad = True
		import os
		if not os.path.exists('models'):
			os.mkdir('models')
			torch.save(models, 'models/model_best_{}_ep{}_{:.4f}.torch'.format(name, cur_epoch, test_f1))

	for param in params:
		param.requires_grad = True

	return max_vali_f1

def get_gnn_embeddings(gnn_model, dataCenter, ds):
    print('Loading embeddings from trained GraphSAGE model.')
    features = np.zeros((len(getattr(dataCenter, ds+'_labels')), gnn_model.out_size))
    nodes = np.arange(len(getattr(dataCenter, ds+'_labels'))).tolist()
    b_sz = 500  # batch size
    batches = math.ceil(len(nodes) / b_sz)
    embs = []
    for index in range(batches):
        nodes_batch = nodes[index*b_sz:(index+1)*b_sz]
        embs_batch = gnn_model(nodes_batch)
        assert len(embs_batch) == len(nodes_batch)
        embs.append(embs_batch)
        # if ((index+1)*b_sz) % 10000 == 0:
        #     print(f'Dealed Nodes [{(index+1)*b_sz}/{len(nodes)}]')
    assert len(embs) == batches
    embs = torch.cat(embs, 0)
    assert len(embs) == len(nodes)
    print('Embeddings loaded.')
    return embs.detach()

def get_gnn_TAXembeddings(gnn_model, dataCenter, ds):
	print('Loading embeddings from trained GraphSAGE model.')
	features = np.zeros((len(getattr(dataCenter, ds+'_labels')), gnn_model.out_size))
	node2ID = getattr(dataCenter, ds+"_node_map")
	ID2node = {id:nsrdzdah for (nsrdzdah, id) in node2ID.items()}
	nodes = np.arange(len(getattr(dataCenter, ds+'_labels'))).tolist()
	b_sz = 500  # batch size
	batches = math.ceil(len(nodes) / b_sz)
	embs = []
	nsrdzdah = []
	for index in range(batches):
		nodes_batch = nodes[index*b_sz:(index+1)*b_sz]
		embs_batch = gnn_model(nodes_batch)
		assert len(embs_batch) == len(nodes_batch)
		embs.append(embs_batch)
		nsrdzdah += [ID2node[n] for n in nodes_batch]  #
		# if ((index+1)*b_sz) % 10000 == 0:
		#     print(f'Dealed Nodes [{(index+1)*b_sz}/{len(nodes)}]')

	assert len(embs) == batches
	embs = torch.cat(embs, 0)
	assert len(embs) == len(nodes) == len(nsrdzdah)
	print('Embeddings loaded.')
	return embs.detach(), nsrdzdah

def train_classification(dataCenter, graphSage, classification, ds, device, max_vali_f1, name, epochs=800, valid_step=5):
	print('Training Classification ...')
	c_optimizer = torch.optim.SGD(classification.parameters(), lr=0.5)
	# train classification, detached from the current graph
	#classification.init_params()
	b_sz = 50
	train_nodes = getattr(dataCenter, ds+'_train')
	labels = getattr(dataCenter, ds+'_labels')
	best_f1 = max_vali_f1
	# features, node_name_list = get_gnn_TAXembeddings(graphSage, dataCenter, ds)
	features = get_gnn_embeddings(graphSage, dataCenter, ds)
	for epoch in range(epochs):
		train_nodes = shuffle(train_nodes)
		batches = math.ceil(len(train_nodes) / b_sz)
		visited_nodes = set()
		for index in range(batches):
			nodes_batch = train_nodes[index*b_sz:(index+1)*b_sz]
			visited_nodes |= set(nodes_batch)
			labels_batch = labels[nodes_batch]
			embs_batch = features[nodes_batch]

			logists = classification(embs_batch)
			loss = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
			loss /= len(nodes_batch)
			# print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(epoch+1, epochs, index, batches, loss.item(), len(visited_nodes), len(train_nodes)))

			loss.backward()
			
			nn.utils.clip_grad_norm_(classification.parameters(), 5)
			c_optimizer.step()
			c_optimizer.zero_grad()
			if index % valid_step == 0:
				metric = validate(dataCenter, ds, graphSage, classification)
				print('validate step [{}/{}] metrics f1 {}, acc {}, precision {}, recall {}'.format(index, batches, metric['f1'], metric['acc'], metric['P'], metric['R']))
				if metric['f1'] > best_f1:
					state = {'graphSage':graphSage.state_dict(), 'classification': classification.state_dict()}
					torch.save(state, './models/{}_graphSage_classification_best_f1.pth'.format(ds))

		# max_vali_f1 = evaluate(dataCenter, ds, graphSage, classification, device, max_vali_f1, name, epoch)
	return classification, best_f1

def apply_model(dataCenter, ds, graphSage, classification, unsupervised_loss, b_sz, unsup_loss, device, learn_method):
	test_nodes = getattr(dataCenter, ds+'_test')
	val_nodes = getattr(dataCenter, ds+'_val')
	train_nodes = getattr(dataCenter, ds+'_train')
	labels = getattr(dataCenter, ds+'_labels')

	if unsup_loss == 'margin':
		num_neg = 6
	elif unsup_loss == 'normal':
		num_neg = 100
	else:
		print("unsup_loss can be only 'margin' or 'normal'.")
		sys.exit(1)

	train_nodes = shuffle(train_nodes)

	models = [graphSage, classification]
	params = []
	for model in models:
		for param in model.parameters():
			if param.requires_grad:
				params.append(param)

	optimizer = torch.optim.SGD(params, lr=0.7)
	optimizer.zero_grad()
	for model in models:
		model.zero_grad()

	batches = math.ceil(len(train_nodes) / b_sz)

	visited_nodes = set()
	for index in range(batches):
		nodes_batch = train_nodes[index*b_sz:(index+1)*b_sz]

		# extend nodes batch for unspervised learning
		# no conflicts with supervised learning
		nodes_batch = np.asarray(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=num_neg)))
		visited_nodes |= set(nodes_batch)

		# get ground-truth for the nodes batch
		labels_batch = labels[nodes_batch]

		# feed nodes batch to the graphSAGE
		# returning the nodes embeddings
		embs_batch = graphSage(nodes_batch)
		# import pdb
		# pdb.set_trace()
		if learn_method == 'sup':
			# superivsed learning
			logists = classification(embs_batch)
			loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
			loss_sup /= len(nodes_batch)
			loss = loss_sup
		elif learn_method == 'plus_unsup':
			# superivsed learning
			logists = classification(embs_batch)
			loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
			loss_sup /= len(nodes_batch)
			# unsuperivsed learning
			if unsup_loss == 'margin':
				loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
			elif unsup_loss == 'normal':
				loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
			loss = loss_sup + loss_net
		else:
			if unsup_loss == 'margin':
				loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
			elif unsup_loss == 'normal':
				loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
			loss = loss_net

		print('{} training Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}]'.format(learn_method, index+1, batches, loss.item(), len(visited_nodes), len(train_nodes)))
		loss.backward()
		for model in models:
			nn.utils.clip_grad_norm_(model.parameters(), 5)
		optimizer.step()

		optimizer.zero_grad()
		for model in models:
			model.zero_grad()

	return graphSage, classification