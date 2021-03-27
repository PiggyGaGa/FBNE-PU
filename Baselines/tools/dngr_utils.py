
# coding: utf-8

"""
Note: Following conventions from 
Tian, F.; Gao, B.; Cui, Q.; Chen, E.; and Liu, T.-Y. 2014. Learning deep representations for graph clustering.
They have 3 types of classifications, a 3 group, 6 group and a 9 group classification.
For each topic in the group, 200 documents are sampled at random. So NG3 contains 600 documents, NG6 contains 1200 documents
and NG9 contains 1800 documents.
Each article is converted into a TF-IDF vector from the whole corpus. 
The graph construction is done by taking the TF-IDF vectors as nodes and the cosine similarity between them as edge weights.

Author : Apoorva Vinod Gorur
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from time import perf_counter
from datetime import timedelta
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn import metrics


def timer(msg):
    def inner(func):
        def wrapper(*args, **kwargs):
            t1 = perf_counter()
            ret = func(*args, **kwargs)
            t2 = perf_counter()
            print("Time elapsed for "+msg+" ----> "+str(timedelta(seconds=t2-t1)))
            print("\n---------------------------------------\n")
            return ret
        return wrapper
    return inner


#Try removing headers, footers and quotes because classifiers tend to overfit and learn only those parts. Remove them
#and let it try learning from body of the documents.
#newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'),categories=categories)

def read_data(group):
    
    if group == 'NG3':
        NG = ['comp.graphics','rec.sport.baseball','talk.politics.guns']
    elif group == 'NG6':
        NG = ['alt.atheism','comp.sys.mac.hardware', 'rec.motorcycles', 'rec.sport.hockey','soc.religion.christian',
             'talk.religion.misc']
    else:
        NG = ['talk.politics.mideast', 'talk.politics.misc', 'comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware',
              'sci.electronics', 'sci.crypt', 'sci.med', 'sci.space', 'misc.forsale']
        
    text_corpus = []
    file_names = []
    target = np.arange(0,len(NG)).tolist()*200
    target.sort()
    for i,category in enumerate(NG):
        np.random.seed(i+42)
        news = fetch_20newsgroups(subset='train',categories=[category])
        permutation = np.arange(len(news.data)).tolist()
        np.random.shuffle(permutation)
        permutation = random.sample(permutation,200)
        randomtext_200 = np.asarray(news.data)[permutation]
        files_200 = news.filenames[permutation]
        text_corpus = text_corpus + randomtext_200.tolist()
        file_names = file_names + files_200.tolist()
        
    return text_corpus, file_names, target



def get_cosine_sim_matrix(text_corpus):
    
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(text_corpus)
    cosine_sim_matrix = cosine_similarity(vectors)
    
    return cosine_sim_matrix



def scale_sim_matrix(mat):
    #Row-wise sacling of matrix
    mat = mat - np.diag(np.diag(mat)) #Make diag elements zero
    rec = np.reciprocal(1e-3 + np.sum(mat, axis=0))
    rec = np.array(rec).flatten()
    #print(rec)
    D_inv = np.diag(rec)
    # sum point is disjoint , add 1e-3 avoid reciprocal 0
    #print(D_inv)
    mat = np.dot(D_inv, mat)   
    return mat



def compute_metrics(embeddings, target):
    
    clf = MultinomialNB(alpha=0.1)
    clf.fit(embeddings, target)
    preds = clf.predict(embeddings)
    f1 = metrics.f1_score(target, preds, average='macro')
    nmi = normalized_mutual_info_score(target,preds)
    
    print("\nEvaluated embeddings using Multinomial Naive Bayes")
    print("F1 - score(Macro) : ",f1)
    print("NMI : ",nmi)
    
    return


@timer("T-SNE visualization")
def visualize_TSNE(embeddings,target):
    tsne = TSNE(n_components=2, init='pca',
                         random_state=0, perplexity=30)
    data = tsne.fit_transform(embeddings)
    #plt.figure(figsize=(12, 6))
    plt.title("TSNE visualization of the embeddings")
    plt.scatter(data[:,0],data[:,1],c=target)

    return

