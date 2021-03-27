import os
import torch
import numpy as np
import random


def to_gpu(gpu, var):
    if gpu and torch.cuda.is_available():
        return var.cuda()
    return var


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.wordcounts = {}

    # to track word counts
    def add_word(self, word):
        if word not in self.wordcounts:
            self.wordcounts[word] = 1
        else:
            self.wordcounts[word] += 1

    def index_word(self):
        vocab_list = [word[0] for word in self.wordcounts.items()]
        index = 0
        for word in vocab_list:
            if word not in self.word2idx:
                self.word2idx[word] = index
                index += 1
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        
    def __len__(self):
        return len(self.word2idx)
def save_txt(list_seq, file):
    with open(file, 'w') as f:
        for seq in list_seq:
            for word in seq:
                f.write(word + ' ')
            f.write('\n')
        f.close()

class Corpus(object):
    def __init__(self, path, maxlen):
        self.dictionary = Dictionary()
        self.maxlen = maxlen
        self.train_path = os.path.join(path, 'train.txt')
        print(self.train_path)

        # make the vocabulary from training set
        self.make_vocab()

        self.train = self.tokenize(self.train_path)

    def make_vocab(self):
        assert os.path.exists(self.train_path)
        # Add words to the dictionary
        with open(self.train_path, 'r') as f:
            for line in f:
                words = line.strip().split(" ")
                for word in words:
                    self.dictionary.add_word(word)
        self.dictionary.index_word()

    def tokenize(self, path):
        """Tokenizes a text file."""
        with open(path, 'r') as f:
            linecount = 0
            lines = []
            for line in f:
                linecount += 1
                words = line.strip().split(" ")
                # vectorize
                vocab = self.dictionary.word2idx
                indices = [vocab[w] for w in words]
                lines.append(indices)
        print("Number of lines in total: {}".format(linecount))
        #print(lines)
        return lines


def random_walk(A_nx, path_length, alpha=0, rand=random.Random(), start=None):
    """ Returns a truncated random walk.
        path_length: Length of the random walk.
        alpha: probability of restarts.
        start: the start node of the random walk.
    """
    if start:
        path = [start]
    else:
        # Sampling is uniform w.r.t Vertex, and not w.r.t Edge
        path = [rand.choice(list(A_nx.nodes))]

    while len(path) < path_length:
        cur = path[-1]
        if len(list(A_nx.neighbors(cur))) > 0:
            if rand.random() >= alpha:
                path.append(rand.choice(list(A_nx.neighbors(cur))))
            else:
                path.append(path[0])
        else:
            break
    #return [str(node-1) for node in path]
    return path


def generate_walks(A_nx, walk_per_node, walk_length, alpha=0, rand=random.Random(0)):
    """
    :param A_nx: ajacency format from networkx package
    :param walk_per_node: number of walks sampled for each node
    :param walk_length: length of walk
    :param alpha: restart probability, 0 means without restart
    :param rand: randomizer
    :return: list of walks rooted from each node
    """
    walks = []
    nodes = list(A_nx.nodes())
    for cnt in range(walk_per_node):
        rand.shuffle(nodes)
        for node in nodes:
            walks.append(random_walk(A_nx, walk_length, rand=rand, alpha=alpha, start=node))
    return walks

def batchify(data, bsz, shuffle=False, gpu=False):
    """
    :param data: training walks sets, that is a list of node walk chain, each chain is a list of nodes
    :param bsz: batch size
    :param shuffle: indicator of if reshuffle training set then split it to batches
    :param gpu: if conduct in gpu
    :return: batches of training samples(walks)
    """
    #print(type(data))
    if shuffle:
        random.shuffle(data)
    nbatch = len(data) // bsz
    batches = []

    for i in range(nbatch):
        # Pad batches to maximum sequence length in batch
        batch = data[i*bsz:(i+1)*bsz]
        # subtract 1 from lengths b/c includes BOTH starts & end symbols
        lengths = [len(x) for x in batch]

        """
        Source and Target for sequence to sequence, here we use autoencoder, thus
        the source and target are the same
        """
        # source has no end symbol
        source = [x for x in batch]
        #print('source', type(source[0][0]))
        # target has no start symbol
        #target = [x for x in batch]

        #for element in target:
        #    print('length source', len(element))
        # find length to pad to
        maxlen = max(lengths)
        for x in source:
            zeros = (maxlen-len(x))*[0]
            x += zeros
        target = source
        source = torch.LongTensor(np.array(source, dtype=np.float))
        target = torch.LongTensor(np.array(target, dtype=np.float)).view(-1)

        if gpu:
            source = source.cuda()
            target = target.cuda()

        batches.append((source, target, lengths))

    return batches


def get_ppl(lm, sentences):
    """
    Assume sentences is a list of strings (space delimited sentences)
    """
    total_nll = 0
    total_wc = 0
    for sent in sentences:
        words = sent.strip().split()
        score = lm.score(sent, bos=True, eos=False)
        word_count = len(words)
        total_wc += word_count
        total_nll += score
    ppl = 10**-(total_nll/total_wc)
    return ppl
