import sys
import os
import torch
import argparse
import pyhocon
import random
import numpy as np
import pandas as pd
from src.TAXDataCenter import TAXDataCenter
from src.utils import evaluate, train_classification, apply_model, get_gnn_TAXembeddings, test, get_gnn_embeddings
from src.models import GraphSage, Classification, UnsupervisedLoss
from src.dataCenter import DataCenter

parser = argparse.ArgumentParser(description='pytorch version of GraphSAGE')

parser.add_argument('--dataSet', type=str, default='TaxH', choices=['TaxH', 'TaxZ', 'TaxS'])
parser.add_argument('--agg_func', type=str, default='MEAN', choices=['MEAN', 'MAX'])
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--b_sz', type=int, default=20)
parser.add_argument('--seed', type=int, default=824)
parser.add_argument('--cuda', action='store_true',
					help='use CUDA')

# parser.add_argument('--cuda', type=str, default='0',
 					# help='use CUDA')

parser.add_argument('--gcn', action='store_true')
parser.add_argument('--learn_method', type=str, default='unsup', choices=['sup', 'unsup', 'plus_unsup'], help='sup： supervised learning, unsup: unsupervised learing, plus_unsup: supervised learning plus unsupervised loss')
parser.add_argument('--unsup_loss', type=str, default='normal', choices=['normal', 'margin'])
parser.add_argument('--max_vali_f1', type=float, default=0)
parser.add_argument('--name', type=str, default='debug')
parser.add_argument('--config', type=str, default='./src/taxConfig.conf')
args = parser.parse_args()

if torch.cuda.is_available():
	if not args.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")
	else:
		device_id = torch.cuda.current_device()
		print('using device', device_id, torch.cuda.get_device_name(device_id))

device = torch.device("cuda:"+args.cuda if args.cuda else "cpu")
print('DEVICE:', device)

if __name__ == '__main__':
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	# # load config file
	config = pyhocon.ConfigFactory.parse_file(args.config)

	# load data
	ds = args.dataSet
	
	basic_feature = '/Data/BasicFeature/{}_normalize.pickle'.format(ds)
	
	label_file = '/Data/Label/{}_label.pickle'.format(ds)
	edges_file = '/Data/NetworkEges/{}_edges.pickle'.format(ds)

	if ds == 'cora':
		dataCenter=DataCenter(config)
		dataCenter.load_dataSet(ds)
	
	else:
		dataCenter = TAXDataCenter(basic_feature, edges_file, label_file)
		dataCenter.load_dataSet(dataSet=ds)

	features = torch.FloatTensor(getattr(dataCenter, ds+'_feats')).to(device)

	graphSage = GraphSage(config['setting.num_layers'], features.size(1), config['setting.hidden_emb_size'], features, getattr(dataCenter, ds+'_adj_lists'), device, gcn=args.gcn, agg_func=args.agg_func)
	graphSage.to(device)

	num_labels = len(set(getattr(dataCenter, ds+'_labels')))
	classification = Classification(config['setting.hidden_emb_size'], num_labels)
	classification.to(device)

	unsupervised_loss = UnsupervisedLoss(getattr(dataCenter, ds+'_adj_lists'), getattr(dataCenter, ds+'_train'), device)

	if args.learn_method == 'sup':
		print('GraphSage with Supervised Learning')
	elif args.learn_method == 'plus_unsup':
		print('GraphSage with Supervised Learning plus Net Unsupervised Learning')
	else:
		print('GraphSage with Net Unsupervised Learning')

	
	# 单纯的训练模型，不进行验证操作
	if args.learn_method == 'unsup':   ## unsupervised 训练
		for epoch in range(args.epochs):
			graphSage, _ = apply_model(dataCenter, ds, graphSage, classification, unsupervised_loss, args.b_sz, args.unsup_loss, device, args.learn_method)
	
	if ds == 'cora':
		embeddings = get_gnn_embeddings(graphSage, dataCenter, ds)
		nsrdzdah = range(embeddings.shape[0])
		embeddings = embeddings.cpu().numpy()
		embeds = pd.DataFrame(embeddings)
		embeds.index = nsrdzdah
		embeddings_fin = embeds
	else:
		embeddings, nsrdzdah = get_gnn_TAXembeddings(graphSage, dataCenter, ds)
		embeddings = embeddings.cpu().numpy()
		embeds = pd.DataFrame(embeddings)
		embeds.index = nsrdzdah
		embeddings_fin = embeds.loc[getattr(dataCenter, ds+'_node_names'), :]
	embeddings_fin.to_pickle('./Embeddings/{}_unsup_embeddings.pickle'.format(ds))

	if args.learn_method == 'unsup':
		classification, args.max_vali_f1 = train_classification(dataCenter, graphSage, classification, ds, device, args.max_vali_f1, args.name, epochs=args.epochs, valid_step=10)
		metric = test(dataCenter, ds, graphSage, classification, checkpoint_name='./models/{}_graphSage_classification_best_f1.pth'.format(ds))
		print('test result metrics f1 {:.4f}, acc {:.4f}, precision {:.4f}, recall {:.4f}'.format(metric['f1'], metric['acc'], metric['P'], metric['R']))
		with open('./result.csv', 'a') as f:
			from datetime import datetime
			now = datetime.now()
			show_time = now.strftime("%m-%d:%H:%M")
			f.write(show_time + '\n')
			f.write('{} result, parameters are batch size {}, if unsupervised {}\n'.format(args.dataSet, args.b_sz, args.learn_method))
			f.write('metrics f1 {:.4f}, acc {:.4f}, precision {:.4f}, recall {:.4f}, auc {:.4f}\n\n\n'.format(metric['f1'], metric['acc'], metric['P'], metric['R'], metric['auc']))