import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tools.Config import Config
import json
import os
import numpy as np
import networkx as nx
from scipy.sparse import csgraph
from tools.netra_utils import to_gpu, Corpus, batchify, generate_walks, save_txt
import torch.optim as optim
import math
import time
import random
import pandas as pd


class MLP_D(nn.Module):
    """
    Discriminator Class using MLP
    """
    def __init__(self, ninput, noutput, layers,
                 activation=nn.LeakyReLU(0.2), gpu=False):
        super(MLP_D, self).__init__()
        self.ninput = ninput
        self.noutput = noutput

        """
            parse network structure
        """
        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        """
            padding different layers, one layer includes liner part, batchNormalization part and activation part
        """
        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)

            # No batch normalization after first layer
            if i != 0:
                bn = nn.BatchNorm1d(layer_sizes[i+1], eps=1e-05, momentum=0.1)
                self.layers.append(bn)
                self.add_module("bn"+str(i+1), bn)

            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        """
            padding output layer
        """
        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = torch.mean(x)
        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass


class MLP_G(nn.Module):
    """
        Generator Class using MLP
    """
    def __init__(self, ninput, noutput, layers,
                 activation=nn.ReLU(), gpu=False):
        super(MLP_G, self).__init__()
        self.ninput = ninput
        self.noutput = noutput

        """
            parse network structure
        """
        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        """
            padding different layers, one layer includes liner part, batchNormalization part and activation part
        """
        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)

            bn = nn.BatchNorm1d(layer_sizes[i+1], eps=1e-05, momentum=0.1)
            self.layers.append(bn)
            self.add_module("bn"+str(i+1), bn)

            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        """
            padding output layer
        """
        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def init_weights(self):
        """
        Initialize model parameters
        :return: no return
        """
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass


class Seq2Seq(nn.Module):
    """
        LSTM autoEncoding
    """
    def __init__(self, emsize, nhidden, ntokens, nlayers, noise_radius=0.2,
                 hidden_init=False, dropout=0, gpu=False):
        super(Seq2Seq, self).__init__()
        self.nhidden = nhidden
        self.emsize = emsize
        self.ntokens = ntokens
        self.nlayers = nlayers
        self.noise_radius = noise_radius
        self.hidden_init = hidden_init
        self.dropout = dropout
        self.gpu = gpu
        self.emb_nhidden = None

        self.start_symbols = to_gpu(gpu, Variable(torch.ones(10, 1).long()))

        # Vocabulary embedding
        self.embedding = nn.Embedding(ntokens, emsize)
        self.embedding_decoder = nn.Embedding(ntokens, emsize)

        # RNN Encoder and Decoder
        self.encoder = nn.LSTM(input_size=emsize,
                               hidden_size=nhidden,
                               num_layers=nlayers,
                               dropout=dropout,
                               batch_first=True)

        decoder_input_size = emsize+nhidden
        self.decoder = nn.LSTM(input_size=decoder_input_size,
                               hidden_size=nhidden,
                               num_layers=1,
                               dropout=dropout,
                               batch_first=True)

        # Initialize Linear Transformation
        self.linear = nn.Linear(nhidden, ntokens)

        self.init_weights()

    def init_weights(self):
        """
        Initializing model weights
        :return:
        """
        initrange = 0.1

        # Initialize Vocabulary Matrix Weight
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding_decoder.weight.data.uniform_(-initrange, initrange)

        # Initialize Encoder and Decoder Weights
        for p in self.encoder.parameters():
            p.data.uniform_(-initrange, initrange)
        for p in self.decoder.parameters():
            p.data.uniform_(-initrange, initrange)

        # Initialize Linear Weight
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)

    def init_hidden(self, bsz):
        """
        Initializing hidden nodes
        :param bsz: batch size
        :return:
        """
        zeros1 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        zeros2 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return (to_gpu(self.gpu, zeros1), to_gpu(self.gpu, zeros2))

    def init_state(self, bsz):
        """
        Initializing state
        :param bsz: batch size
        :return:
        """
        zeros = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return to_gpu(self.gpu, zeros)

    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

#return embedding of current batch

    def return_emb(self):
        return self.emb_nhidden

    def embed_dictionary(self, dic):
        embeddings = self.embedding(dic)
        return embeddings

    def embed_after_LSTM(self, dic, length):
        embeddings = self.forward(dic, length, noise = False)
        return self.emb_nhidden

    def forward(self, indices, lengths, noise, encode_only=False):
        """
        Given nodes, feedforward to get the embedding codes of nodes
        :param indices: node id's
        :param lengths: length of walk
        :param noise:
        :param encode_only:
        :return:
        """
        batch_size, maxlen = indices.size()

        hidden = self.encode(indices, lengths, noise)

        if encode_only:
            return hidden

        if hidden.requires_grad:
            hidden.register_hook(self.store_grad_norm)

        decoded = self.decode(hidden, batch_size, maxlen,
                              indices=indices, lengths=lengths)
        return decoded

    def encode(self, indices, lengths, noise):
        embeddings = self.embedding(indices)
        packed_embeddings = pack_padded_sequence(input=embeddings,
                                                 lengths=lengths,
                                                 batch_first=True,
                                                 enforce_sorted=False)

        # Encode
        packed_output, state = self.encoder(packed_embeddings)

        hidden, cell = state
        # batch_size x nhidden
        hidden = hidden[-1]  # get hidden state of last layer of encoder

        # normalize to unit ball (l2 norm of 1) - p=2, dim=1
        hidden = hidden / torch.norm(hidden, p=2, dim=1, keepdim=True)

        # norms = torch.norm(hidden, 2, 1)
        # hidden = torch.div(hidden, norms.expand_as(hidden))

        if noise and self.noise_radius > 0:
            #print(hidden.size())
            gauss_noise = torch.normal(mean=torch.zeros(hidden.size()),
                                       std=self.noise_radius)
            hidden = hidden + to_gpu(self.gpu, Variable(gauss_noise))

        return hidden

    def decode(self, hidden, batch_size, maxlen, indices=None, lengths=None):
        # batch x hidden
        all_hidden = hidden.unsqueeze(1).repeat(1, maxlen, 1)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        embeddings = self.embedding_decoder(indices)
        augmented_embeddings = torch.cat([embeddings, all_hidden], 2)
        packed_embeddings = pack_padded_sequence(input=augmented_embeddings,
                                                 lengths=lengths,
                                                 batch_first=True,
                                                 enforce_sorted=False)

        packed_output, state = self.decoder(packed_embeddings, state)
        output, lengths = pad_packed_sequence(packed_output, batch_first=True)

        # reshape to batch_size*maxlen x nhidden before linear over vocab
        decoded = self.linear(output.contiguous().view(-1, self.nhidden))
        decoded = decoded.view(batch_size, maxlen, self.ntokens)

        # ADD this can be used as embedding
        self.emb_nhidden = output.contiguous().view(-1, self.nhidden)

        return decoded


def load_models(load_path):
    """
    Load models save in self.config.json file and word vocabulary in vocab.json
    :param load_path: fold to store self.config.json and vocab.json files
    :return: model hyper-parameters, word and index, AE model, GAN-generator model, GAN-discriminator model
    """
    model_args = json.load(open("{}/self.config.json".format(load_path), "r"))
    word2idx = json.load(open("{}/vocab.json".format(load_path), "r"))
    idx2word = {v: k for k, v in word2idx.items()}

    autoencoder = Seq2Seq(emsize=model_args['emsize'],
                          nhidden=model_args['nhidden'],
                          ntokens=model_args['ntokens'],
                          nlayers=model_args['nlayers'],
                          hidden_init=model_args['hidden_init'])
    gan_gen = MLP_G(ninput=model_args['z_size'],
                    noutput=model_args['nhidden'],
                    layers=model_args['arch_g'])
    gan_disc = MLP_D(ninput=model_args['nhidden'],
                     noutput=1,
                     layers=model_args['arch_d'])

    print('Loading models from'+load_path)
    ae_path = os.path.join(load_path, "autoencoder_model.pt")
    gen_path = os.path.join(load_path, "gan_gen_model.pt")
    disc_path = os.path.join(load_path, "gan_disc_model.pt")

    autoencoder.load_state_dict(torch.load(ae_path))
    gan_gen.load_state_dict(torch.load(gen_path))
    gan_disc.load_state_dict(torch.load(disc_path))
    return model_args, idx2word, autoencoder, gan_gen, gan_disc


class NetRA(object):
    def __init__(self, graph, config = None):
        if config:
            self.config = config
        else:
            config_dict = {
                'outf' : 'example',
                'seed' : 1111,
                'maxlen' : 100,   # walk to generating walks
                'nhidden' : 128,   # dimension of embeddings
                'emsize' : 30,  
                'nlayers' : 1,
                'noise_radius' : 0.2,
                'noise_anneal' : 0.995,
                'hidden_init' : 1,
                'arch_g' : '300-300',
                'arch_d' : '300-300',
                'cuda' : 1, 
                'numWalks_per_node' : 30,
                'batch_size' : 256,  #  64 before
                'dropout' : 0.0, 
                'z_size' : 100, 
                'lr_ae' : 1,
                'lr_gan_g' : 5e-5,
                'lr_gan_d' : 1e-5,
                'belta1' : 0.9,
                'epochs' : 2,
                'niters_ae' : 1,
                'niters_gan_d' : 5,
                'niters_gan_g' : 1,
                'niters_gan_schedule' : '2-4-6-10-20-30-40',
                'min_epochs' : 6,
                'no_earlystopping' : 1,
                'clip' : 1,
                'gan_clamp' : 0.01,
                'sample' : 1,
                'log_interval' : 200,
                'temp' : 1,
                'enc_grad_norm' : 1,
                'gan_toenc' : 0.01,
                'walk_length' : 20,
                'beta1' : 0.9

            }
            self.config = Config(config_dict)
        # set the random seed manually

        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            if not self.config.cuda:
                print("WARNING: You have a CUDA device, "
                    "so you should probably run with --cuda")
            else:
                torch.cuda.manual_seed(self.config.seed)
        # load data
        A = nx.to_scipy_sparse_matrix(graph)
        L = csgraph.laplacian(A, normed=False)       
        L = np.array(L.toarray(), np.float32)
        L = Variable(torch.from_numpy(L))
        self.L = to_gpu(self.config.cuda, L)
        # generate walk for each node with given walk_length
        walks = generate_walks(graph, self.config.numWalks_per_node, self.config.walk_length)
        save_txt(walks,'./corpus/train.txt')
        self.corpus = Corpus('./corpus/', maxlen=self.config.maxlen)

        self.train_data = batchify(self.corpus.train, self.config.batch_size, shuffle=True)

        self.EMBED_SEGMENT = 4000
        ## build models
        self.ntokens = len(self.corpus.dictionary.word2idx)
        self.autoencoder = Seq2Seq(emsize=self.config.emsize,
                            nhidden=self.config.nhidden,
                            ntokens=self.ntokens,
                            nlayers=self.config.nlayers,
                            noise_radius=self.config.noise_radius,
                            hidden_init=self.config.hidden_init,
                            dropout=self.config.dropout,
                            gpu=self.config.cuda)

        self.gan_gen = MLP_G(ninput=self.config.z_size, noutput=self.config.nhidden, layers=self.config.arch_g)
        self.gan_disc = MLP_D(ninput=self.config.nhidden, noutput=1, layers=self.config.arch_d)
        print(self.autoencoder)
        print(self.gan_gen)
        print(self.gan_disc)


    def train(self):
        # to cuda if cuda is available
        data_loader = self.train_data
        self.autoencoder = to_gpu(self.config.cuda, self.autoencoder)
        self.gan_gen = to_gpu(self.config.cuda, self.gan_gen)
        self.gan_disc = to_gpu(self.config.cuda, self.gan_disc)

        # optimizer
        self.optimizer_ae = optim.SGD(self.autoencoder.parameters(), lr=self.config.lr_ae)
        self.optimizer_gan_g = optim.Adam(self.gan_gen.parameters(),
                                    lr=self.config.lr_gan_g,
                                    betas=(self.config.beta1, 0.999))
        self.optimizer_gan_d = optim.Adam(self.gan_disc.parameters(),
                                    lr=self.config.lr_gan_d,
                                    betas=(self.config.beta1, 0.999))
        criterion_ce = nn.CrossEntropyLoss()


        # schedule of increasing GAN training loops
        if self.config.niters_gan_schedule != "":
            gan_schedule = [int(x) for x in self.config.niters_gan_schedule.split("-")]
        else:
            gan_schedule = []
        niter_gan = 1   # start from 1, and will be dynamically increased

        fixed_noise = to_gpu(self.config.cuda,
                            Variable(torch.ones(self.config.batch_size, self.config.z_size)))
        fixed_noise.data.normal_(0, 1)
        

        for epoch in range(1, self.config.epochs+1):
            #self.embed_afterLSTM(self.corpus, self.config.nhidden)
            print('The {} th epoch'.format(epoch))
            # update gan training schedule  gan schedule 设置的是什么时候训练 gan 网络
            if epoch in gan_schedule:
                niter_gan += 1
                print("GAN training loop schedule increased to {}".format(niter_gan))
                with open("./output/logs.txt", 'a') as f:
                    f.write("GAN training loop schedule increased to {}\n".
                            format(niter_gan))
            total_loss_ae = 0
            epoch_start_time = time.time()
            start_time = time.time()
            niter = 0
            niter_global = 1
            # loop through all batches in training data
            while niter < len(data_loader):
                """ 
                    Iteratively conduct autoencoder training, then GAN regularization,
                    The GAN part includes discriminator and generator iteratively.
                """
                # train autoencoder ----------------------------
                for i in range(self.config.niters_ae):
                    if niter == len(data_loader):
                        break  # end of epoch
                    total_loss_ae, start_time = self.train_ae(data_loader[niter], total_loss_ae, start_time, niter, criterion_ce, epoch)
                    niter += 1

                # train gan ----------------------------------
                for k in range(niter_gan):

                    # train discriminator/critic
                    for i in range(self.config.niters_gan_d):
                        # feed a seen sample within this epoch; good for early training
                        errD, errD_real, errD_fake = self.train_gan_d(data_loader[random.randint(0, len(data_loader)-1)])

                    # train generator
                    for i in range(self.config.niters_gan_g):
                        errG = self.train_gan_g()

                """
                    The codes here are for logging running status, not actually conduct the algorithm logic
                """
                niter_global += 1
                if niter_global % 100 == 0:
                    print('[%d/%d][%d/%d] Loss_D: %.8f (Loss_D_real: %.8f '
                        'Loss_D_fake: %.8f) Loss_G: %.8f'
                        % (epoch, self.config.epochs, niter, len(data_loader),
                            errD.item(), errD_real.item(),
                            errD_fake.item(), errG.item()))
                    with open("./output/{}/logs.txt".format(self.config.outf), 'a') as f:
                        f.write('[%d/%d][%d/%d] Loss_D: %.8f (Loss_D_real: %.8f '
                                'Loss_D_fake: %.8f) Loss_G: %.8f\n'
                                % (epoch, self.config.epochs, niter, len(data_loader),
                                errD.item(), errD_real.item(),
                                errD_fake.item(), errG.item()))
                    # exponentially decaying noise on autoencoder
                    self.autoencoder.noise_radius = self.autoencoder.noise_radius*self.config.noise_anneal
            # embed_corpus(corpus, self.config.emsize)
            self.embed_afterLSTM(self.corpus, self.config.nhidden)
            # save_model()
            # shuffle between epochs
            data_loader = batchify(self.corpus.train, self.config.batch_size, shuffle=True)
        return                

    def embed_afterLSTM(self, corpus, emsize):
        """
        Getting embedding codes
        :param corpus: nodes of graph
        :param emsize: number of dimensions of embedded codes
        :return: embedding vectors
        """
        dic = list(corpus.dictionary.word2idx.values())
        dic = np.sort(dic)
        if self.ntokens <= self.EMBED_SEGMENT:
            dic = np.vstack((dic,dic))
            dic_to_embed = Variable(torch.from_numpy(dic))
            dic_to_embed = to_gpu(self.config.cuda, dic_to_embed)
            embeded = self.autoencoder.embed_after_LSTM(dic_to_embed, [self.ntokens, self.ntokens])
            #print('embeded', embeded[0])
            dic_vector = embeded[:self.ntokens].cpu().data.numpy()
        else:
            dic_i = np.array_split(dic, self.ntokens//self.EMBED_SEGMENT)
            dic_1 = dic_i[0]
            n_dic_1 = len(dic_1)
            dic_1 = np.vstack((dic_1,dic_1))
            dic_to_embed = Variable(torch.from_numpy(dic_1))
            dic_to_embed = to_gpu(self.config.cuda, dic_to_embed)
            embeded = self.autoencoder.embed_after_LSTM(dic_to_embed, [n_dic_1,n_dic_1])
            print('embeded', embeded[0])
            dic_vector = embeded[:n_dic_1].cpu().data.numpy()
            for j in range(1, self.ntokens//self.EMBED_SEGMENT):
                dic_j = dic_i[j]
                n_dic_j = len(dic_j)
                dic_j = np.vstack((dic_j,dic_j))
                dic_to_embed = Variable(torch.from_numpy(dic_j))
                dic_to_embed = to_gpu(self.config.cuda, dic_to_embed)
                embeded = self.autoencoder.embed_after_LSTM(dic_to_embed, [n_dic_j,n_dic_j])
                dic_vector_j = embeded[:n_dic_j].cpu().data.numpy()
                dic_vector = np.vstack((dic_vector, dic_vector_j))
            dic = np.vstack((dic,dic))
                
        dic_tosave = np.insert(dic_vector, 0, dic[0], axis=1)
        np.savetxt('./output/embed_afterLSTM.txt', dic_tosave,
                fmt=' '.join(['%i'] + ['%1.6f'] * emsize))
        return dic_vector, corpus.dictionary.idx2word

    def embed_dic_for_loss(self, corpus, emsize):
        dic = list(corpus.dictionary.word2idx.values())
        #print(len(dic))
        dic = np.sort(np.array(dic), axis=None)
        if self.ntokens <= self.EMBED_SEGMENT:
            dic = np.vstack((dic,dic))
            dic_to_embed = Variable(torch.from_numpy(dic))
            dic_to_embed = to_gpu(self.config.cuda, dic_to_embed)
            embeded = self.autoencoder.embed_after_LSTM(dic_to_embed, [self.ntokens, self.ntokens])
            dic_vector = embeded[:self.ntokens]
        else:
            dic_i = np.array_split(dic, self.ntokens/self.EMBED_SEGMENT)
            dic_1 = dic_i[0]
            #print(dic_1)
            n_dic_1 = len(dic_1)
            dic_1 = np.vstack((dic_1,dic_1))
            dic_to_embed = Variable(torch.from_numpy(dic_1))
            dic_to_embed = to_gpu(self.config.cuda, dic_to_embed)
            embeded = self.autoencoder.embed_after_LSTM(dic_to_embed, [n_dic_1,n_dic_1])
            dic_vector = embeded[:n_dic_1]
            for j in range(1, self.ntokens//self.EMBED_SEGMENT):
                dic_j = dic_i[j]
                n_dic_j = len(dic_j)
                dic_j = np.vstack((dic_j,dic_j))
                dic_to_embed = Variable(torch.from_numpy(dic_j))
                dic_to_embed = to_gpu(self.config.cuda, dic_to_embed)
                embeded = self.autoencoder.embed_after_LSTM(dic_to_embed, [n_dic_j,n_dic_j])
                dic_vector_j = embeded[:n_dic_j]
                dic_vector = torch.cat((dic_vector, dic_vector_j), 0)
        return dic_vector


    def unique(self, tensor1d):
        t, idx = np.unique(tensor1d.cpu().data.numpy(), return_index=True)
        return t, idx


    def train_ae(self, batch, total_loss_ae, start_time, i, criterion_ce, epoch):
        """
        Training LSTM AE
        :param batch: one batch of data
        :param total_loss_ae: accumulated loss for LSTM AE so far
        :param start_time: for timming
        :param i: current iteration ID
        :return: accumulated total loss of LSTM AE part so far, and start time for timing
        """
        self.autoencoder.train()
        self.autoencoder.zero_grad()

        source, target, lengths = batch
        source = to_gpu(self.config.cuda, Variable(source))
        target = to_gpu(self.config.cuda, Variable(target))

        # Create sentence length mask over padding
        mask = target.gt(0)
        masked_target = target.masked_select(mask)
        # examples x ntokens
        output_mask = mask.unsqueeze(1).expand(mask.size(0), self.ntokens)

        # output: batch x seq_len x ntokens
        output = self.autoencoder(source, lengths, noise=True)

        # output_size: batch_size, maxlen, self.ntokens
        flattened_output = output.view(-1, self.ntokens)


        emb_py = self.embed_dic_for_loss(self.corpus, self.config.nhidden)
        embT = torch.transpose(emb_py,0,1)
        embT = torch.mm(embT, self.L)

        adj_loss = torch.trace(torch.mm(embT, emb_py)) / self.ntokens

        masked_output = flattened_output.masked_select(output_mask).view(-1, self.ntokens)
        loss = criterion_ce(masked_output/self.config.temp, masked_target)
        print('loss', loss.data)

        loss += adj_loss

        loss.backward()

        # `clip_grad_norm` to prevent exploding gradient in RNNs / LSTMs
        # This is the version of Wasserstein GAN, which has gradient clipping
        torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), self.config.clip)
        self.optimizer_ae.step()

        total_loss_ae += loss.data

        accuracy = None

        ######################## store log periodically ############################
        if i % self.config.log_interval == 0 and i > 0:
            # accuracy
            probs = F.softmax(masked_output, dim=0)
            max_vals, max_indices = torch.max(probs, 1)
            accuracy = torch.mean(max_indices.eq(masked_target).float()).item()

            cur_loss = total_loss_ae.item() / self.config.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                'loss {:5.2f} | ppl {:8.2f} | acc {:8.2f}'.format(epoch, i, len(self.train_data),
                        elapsed * 1000 / self.config.log_interval,
                        cur_loss, math.exp(cur_loss), accuracy))
            total_loss_ae = 0
            start_time = time.time()
        return total_loss_ae, start_time


    def train_gan_g(self,):
        """
        Training WGAN generator network
        :return: error of generator part
        """
        self.gan_gen.train()
        self.gan_gen.zero_grad()

        noise = to_gpu(self.config.cuda,
                    Variable(torch.ones(self.config.batch_size, self.config.z_size)))
        noise.data.normal_(0, 1)

        fake_hidden = self.gan_gen(noise)
        errG = self.gan_disc(fake_hidden)

        # loss / backprop
        one = to_gpu(self.config.cuda, torch.FloatTensor([1]))
        errG.backward(one)
        self.optimizer_gan_g.step()

        return errG

    def grad_hook(self, grad):
        # Gradient norm: regularize to be same
        # code_grad_gan * code_grad_ae / norm(code_grad_gan)
        if self.config.enc_grad_norm:
            gan_norm = torch.norm(grad, 2, 1).detach().data.mean()
            normed_grad = grad * self.autoencoder.grad_norm / gan_norm
        else:
            normed_grad = grad

        # weight factor and sign flip
        normed_grad *= -math.fabs(self.config.gan_toenc)
        return normed_grad


    def train_gan_d(self, batch):
        """
        Training WGAN discriminator
        :param batch: training batch data
        :return: discriminator part error
        """
        # clamp parameters to a cube
        # WGAN Weight clipping
        for p in self.gan_disc.parameters():
            p.data.clamp_(-self.config.gan_clamp, self.config.gan_clamp)

        self.autoencoder.train()
        self.autoencoder.zero_grad()
        self.gan_disc.train()
        self.gan_disc.zero_grad()

        # positive samples ----------------------------
        # generate real codes
        source, target, lengths = batch
        source = to_gpu(self.config.cuda, Variable(source))
        target = to_gpu(self.config.cuda, Variable(target))

        # batch_size x nhidden
        real_hidden = self.autoencoder(source, lengths, noise=False, encode_only=True)
        real_hidden.register_hook(self.grad_hook)

        # loss / backprop
        errD_real = self.gan_disc(real_hidden)
        one = to_gpu(self.config.cuda, torch.FloatTensor([1]))
        errD_real.backward(one)

        # negative samples ----------------------------
        # generate fake codes
        noise = to_gpu(self.config.cuda,
                    Variable(torch.ones(self.config.batch_size, self.config.z_size)))
        noise.data.normal_(0, 1)

        # loss / backprop
        fake_hidden = self.gan_gen(noise)
        errD_fake = self.gan_disc(fake_hidden.detach())
        one = to_gpu(self.config.cuda, torch.FloatTensor([1]))
        mone = one * -1
        errD_fake.backward(mone)

        # `clip_grad_norm` to prvent exploding gradient problem in RNNs / LSTMs
        # This is the version of Wasserstein GAN, which has gradient clipping
        torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), self.config.clip)

        self.optimizer_gan_d.step()
        self.optimizer_ae.step()
        errD = -(errD_real - errD_fake)
        return errD, errD_real, errD_fake

    def save_embeddings(self, file):
        data = self.get_embeddings()
        data.to_pickle(file)
        
    def get_embeddings(self):
        embeddings, index2word = self.embed_afterLSTM(self.corpus, self.config.nhidden)
        index = 0
        embeddings_dict = {}
        for embed in embeddings:
            embeddings_dict[index2word[index]] = embed
            index += 1
        embedding_pd = pd.DataFrame.from_dict(embeddings_dict, orient='index')
        print(embedding_pd.shape[0])
        return embedding_pd
        