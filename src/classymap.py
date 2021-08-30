import embeddings
from cupy_utils import *
import cupyx
import os
import argparse
import collections
import numpy as np
import sys
from scipy.sparse import csr_matrix, hstack as sparse_hstack
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle
import time
from random import randrange
from art_wrapper import run_supervised_alignment
from sklearn.utils import shuffle
import math
from wordfreq import word_frequency
import textdistance
from sklearn.model_selection import cross_val_score
from earthmover import earthmover_distance
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
import random
from bpemb import BPEmb
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPClassifier

# src list of indexes from the source vocab to get neighbours for (these index the rows of X)
# X embedding matrix for the source language 
# Z embedding matrix for the target language
# num_NN number of neighbours to generate
# returns a dictionary int --> list(int), each value is a list of length k containing the indexes (from Z) of the nearest neighbours (sorted descending by similarity)
# should be pretty fast on GPU with cuda = "True"
# mode can be "dense" or "sparse"
# supply return_scores = False to make it faster (and you dont need scores)
def get_NN(src, X, Z, num_NN, cuda = False, mode = "dense", batch_size = 100, return_scores = True, fast1NNmode = False):
  # get Z to the GPU once in the beginning (it can be big, seems like a waste to copy it again for every batch)
  if cuda:
    if not supports_cupy():
      print("Error: Install CuPy for CUDA support", file=sys.stderr)
      sys.exit(-1)
    xp = get_cupy()
    if mode == "dense":
      z = xp.asarray(Z)
    elif mode == "sparse":
      z = cupyx.scipy.sparse.csr_matrix(Z)
  else: # not cuda
    z = Z
 
  ret = {}
  for i in range(0, len(src), batch_size):
    print("Starting NN batch " + str(i/batch_size))
    start_time = time.time()

    j = min(i + batch_size, len(src))    
    x_batch_slice = X[src[i:j]]
    
    # get the x part to the GPU if needed
    if cuda:
     if mode == "dense":
        x_batch_slice = xp.asarray(x_batch_slice)
     elif mode == "sparse":
        x_batch_slice = cupyx.scipy.sparse.csr_matrix(x_batch_slice)
    
    similarities = x_batch_slice.dot(z.T)

    if mode == "sparse":
      similarities = similarities.todense()

    nn = (-similarities).argsort(axis=1)

    for k in range(j-i):
      ind = nn[k,0:num_NN].tolist()
      if mode == "sparse" and not cuda:
        ind = ind[0]

      if return_scores:             
        sim = similarities[k,ind].tolist()
        ret[src[i+k]] = list(zip(ind,sim))
      else:
        ret[src[i+k]] = ind
    print("Time taken " + str(time.time() - start_time))
  return(ret)

def get_1NNfast(src, X, Z, cuda = False, batch_size = 100, return_scores = True):
  # get Z to the GPU once in the beginning (it can be big, seems like a waste to copy it again for every batch)
  if cuda:
    if not supports_cupy():
      print("Error: Install CuPy for CUDA support", file=sys.stderr)
      sys.exit(-1)
    xp = get_cupy()
    z = xp.asarray(Z)
  else: # not cuda
    z = Z
 
  nn_list = []
  sims_list = []
  for i in range(0, len(src), batch_size):
    start_time = time.time()

    j = min(i + batch_size, len(src))    
    x_batch_slice = X[src[i:j]]
    
    # get the x part to the GPU if needed
    if cuda:
      x_batch_slice = xp.asarray(x_batch_slice)
    
    similarities = x_batch_slice.dot(z.T)
    nn = (similarities).argmax(axis=1)
    nn_list += nn.tolist()
    if return_scores:
      sims_list += similarities[np.array(range(similarities.shape[0])),nn].tolist()

  if return_scores:
    return dict(zip(src, zip(nn_list, sims_list)))
  else:
    return dict(zip(src, nn_list))
       

# for each word in src_words computes the num_NN most similar by char ngram cosine, these are the candidate pairs (a total of len(src_words) x num_NN of them)
# returns the candidate pairs sorted in descending order (but not all of them because this is still too much, rather num_output top candidates from the sorted list)
# optionally a resort function can be supplied which will sort (descending) the top num_output candidates one more time (for example if we wanted to sort the top
# examples by edit distance instead of ngram cosine)

def precompute_orthographic_NN(src_words, tar_words, num_NN, num_output = 20000, cuda = False, resort_func = None):
  ret = {}
  tid = TfidfVectorizer(analyzer = "char", min_df = 1, ngram_range = (2,5))
  print("Generating the char n-gram vectors")
  v = tid.fit_transform(src_words + tar_words)
  src_vectors = v[0:len(src_words),:]
  tar_vectors = v[len(src_words):,:]
  
  freq_cutoff = 0.10
  firstN = int(freq_cutoff * len(src_words))
  tar_vectors = tar_vectors[0:firstN,:] # leave only 10% most frequent target words

  src_ind2w = {i: word for i, word in enumerate(src_words)}
  tar_ind2w = {i: word for i, word in enumerate(tar_words)}
 
  print("Sparse matrix multiplication")
  d = get_NN(range(len(src_words)), src_vectors, tar_vectors, num_NN, cuda = cuda, mode = "sparse", batch_size = 1000) # batch of 500 fits on a GPU with 8G RAM
  print("Finished!")

  print("Sorting candidates and finishing up")
  s = []
  for sw in d:
    for tw,c in d[sw]:
        s.append((sw,tw,c))
 
  sorted_s = sorted(s, key = lambda t:t[2], reverse = True) # sort descending by cosine sim
  sorted_s = sorted_s[0:num_output]
  if resort_func is not None:
    sorted_s = sorted(sorted_s, key = lambda t:resort_func(t[0],t[1]), reverse = True)
  ret = [(src_ind2w[k], tar_ind2w[v], c) for k,v,c in sorted_s]
  return(ret, d)


class BertRepresentationGenerator():
  def __init__(self):
    pretrained_weights = 'bert-base-multilingual-cased'
    self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
    self.model = BertModel.from_pretrained(pretrained_weights)
    self.model = self.model.to("cuda")
    self.dict = {}
  
  def generate_representation(self, word):
    input_ids = torch.tensor([self.tokenizer.encode(word, add_special_tokens = True)])
    input_ids = input_ids.to("cuda")
    with torch.no_grad():
      last_hidden_states = self.model(input_ids)[0]
    return last_hidden_states.cpu()

  def precompute_to_file(self, words, lang): # mode is src or tar
    outfilename = "bert-" + lang + ".pickle"
    d = {}
    for w in tqdm(words):
      d[w] = self.generate_representation(w)
    with open(outfilename, "wb") as outfile: 
      pickle.dump(d, outfile)
    self.dict[lang] = d
  

  def load_from_file(self, lang):
    infilename = "bert-" + lang + ".pickle"
    with(open(infilename, "rb")) as infile:
      d = pickle.load(infile)
    self.dict[lang] = d

  def generate_representation_fast(self, word, lang):
    return self.dict[lang][word]

 

class FeatureCalculator():

  def __init__(self, src_words, tar_words, src_embeddings, tar_embeddings, src_code, tar_code, ornn, args):
    #self.char_tfidf = TfidfVectorizer(analyzer = "char", min_df = 1, ngram_range = (2,5)).fit(src_words + tar_words)
    self.src_w2ind = {word: i for i, word in enumerate(src_words)}
    self.tar_w2ind = {word: i for i, word in enumerate(tar_words)}
    self.x = src_embeddings
    self.z = tar_embeddings
    self.src_code = src_code
    self.tar_code = tar_code
    self.bert = BertRepresentationGenerator()
    
    self.pca_s = PCA(n_components = 10)
    self.pca_t = PCA(n_components = 10)

    self.x_pca = self.pca_s.fit_transform(cupy.asnumpy(self.x))
    self.z_pca = self.pca_t.fit_transform(cupy.asnumpy(self.z))
    self.ornn = ornn
    self.src_count_vect = CountVectorizer(ngram_range = (1,4), analyzer = "char")
    self.tar_count_vect = CountVectorizer(ngram_range = (1,4), analyzer = "char")
    #self.interaction = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False) 
    self.kbest = SelectKBest(chi2, k = 10)
    self.args = args
    self.multibpemb = BPEmb(lang="multi", vs=1000000, dim=300) # mostly returns 1 bpe per word, weird, TODO investigate

  def update_embeddings(self, src_embeddings, tar_embeddings):
    self.x = src_embeddings
    self.z = tar_embeddings
    self.x_pca = self.pca_s.fit_transform(cupy.asnumpy(self.x))
    self.z_pca = self.pca_t.fit_transform(cupy.asnumpy(self.z))
    
  
  def calc_features(self, word_pairs, labels, mode = "train"): # word_pairs should be a list of tuples and x and z numpy matrices representing the currently aligned embeddings
    features = []
    
    if self.args.use_char_ngrams == 1:
      src_words = [wp[0] for wp in word_pairs]
      tar_words = [wp[1] for wp in word_pairs]
      if mode == "train":
        src_char_feats = self.src_count_vect.fit_transform(src_words)
        tar_char_feats = self.tar_count_vect.fit_transform(tar_words)
        print(src_char_feats.shape)
        print(tar_char_feats.shape)
        all_char_feats = sparse_hstack((src_char_feats, tar_char_feats))
        #all_char_feats_inter  = self.interaction.fit_transform(all_char_feats)
        all_char_feats_inter  = all_char_feats # no interaction, too slow
        best_char_feats = self.kbest.fit_transform(all_char_feats_inter, labels)
      elif mode == "test":
        src_char_feats = self.src_count_vect.transform(src_words)
        tar_char_feats = self.tar_count_vect.transform(tar_words)
        all_char_feats = sparse_hstack((src_char_feats, tar_char_feats))
        #all_char_feats_inter = self.interaction.transform(all_char_feats)
        all_char_feats_inter = all_char_feats # no interaction, too slow
        best_char_feats = self.kbest.transform(all_char_feats_inter)
      
      best_char_feats = best_char_feats.todense()
      for i in range(best_char_feats.shape[1]):
        features.append(list([x[0] for x in best_char_feats[:,i].tolist()]))

    if self.args.use_edit_dist == 1:
      #print("Levensthein edit distance feats ...")
      # Levensthein edit distance
      levenshtein_feats = [textdistance.levenshtein.distance(sw, tw) for sw, tw in word_pairs]
      features.append(levenshtein_feats)
    
      #print("Jaro Winkler edit distance feats ...")
      # Jaro Winkler edit distance
      jaro_feats = [textdistance.jaro.distance(sw, tw) for sw, tw in word_pairs]
      features.append(jaro_feats)
      
      norm_lev_feats = [textdistance.levenshtein.distance(sw, tw)/np.mean([float(len(sw)), float(len(tw))]) for sw, tw in word_pairs]
      features.append(norm_lev_feats)
      
      rank_lev_feats = [math.log(textdistance.levenshtein.distance(sw, tw) + 1) for sw, tw in word_pairs]
      features.append(rank_lev_feats)    
  
      edit_combo_feats = [norm_lev_feats[i] + rank_lev_feats[i] for i in range(len(norm_lev_feats))]    
      features.append(edit_combo_feats)    

    if self.args.use_aligned_cosine == 1:
      #print("Aligned embeddings cosine feats ...")
      # cosine sim in the aligned embedding space
      aligned_cosine_feats = []
      for sw, tw in word_pairs:
        swv = self.x[self.src_w2ind[sw],:]
        twv = self.z[self.tar_w2ind[tw],:]
        aligned_cosine_feats.append(swv.dot(twv.T).item())
      features.append(aligned_cosine_feats)

    if self.args.use_aligned_pca == 1:
      src_pca_feats, tar_pca_feats = {}, {}     
      for sw,tw in word_pairs:
        swv = self.x_pca[self.src_w2ind[sw],:]
        twv = self.z_pca[self.tar_w2ind[tw],:]
        for i in range(swv.shape[0]):
          if i not in src_pca_feats:
            src_pca_feats[i] = []
          src_pca_feats[i].append(swv[i])
        for i in range(twv.shape[0]):
          if i not in tar_pca_feats:
            tar_pca_feats[i] = []
          tar_pca_feats[i].append(twv[i])
      for k in src_pca_feats:
        features.append(src_pca_feats[k])
      for k in tar_pca_feats:
        features.append(tar_pca_feats[k])


    # n-gram overlap
    def find_ngrams(input_list, n):
       return list(zip(*[input_list[i:] for i in range(n)]))
    
    if self.args.use_ngrams == 1:
      for n in range(2, 6):
        cngofeats_n = [] 
        for sw, tw in word_pairs:
          cns = find_ngrams([c for c in sw], n)
          cnt = find_ngrams([c for c in tw], n)
          s1, s2, s1s2 = len(cns), len(cnt), len ([x for x in cns if x in cnt])
        
          cngofeats_n.append((2 / (s1/s1s2 + s2/s1s2)) if s1s2 != 0 else 0)
        features.append(cngofeats_n)
      
    if self.args.use_full_bert == 1:
      # BERT feats
      bert_avg_feats = []
      bert_pwmax_feats = []
      bert_earthmovers_feats = []
      bert_cls_feats = []

      print("Calculating BERT feats ...")
      for sw, tw in word_pairs:
        start = time.time()
        swb = self.bert.generate_representation(sw)
        twb = self.bert.generate_representation(tw)      
        #print("Finished with bert")
        #print(time.time() - start)

        # cosine of averaged wordpiece embeddings
        swb_avg = swb.mean(1)
        twb_avg = twb.mean(1)
        bert_avg_feats.append(cosine_similarity(swb_avg, twb_avg)[0][0])
        #print("Finished with avg")
        #print(time.time() - start)

        # pairwise max cosine of wordpiece embeddings
        pairwise_sims = cosine_similarity(swb[0], twb[0])
        bert_pwmax_feats.append(pairwise_sims.max())
        #print("Finished with pairwisemax")
        #print(time.time() - start)
      
        # earthmovers distance 
        swbe = [tuple(swb[0,i,:].tolist()) for i in range(swb.shape[1])]
        twbe = [tuple(twb[0,i,:].tolist()) for i in range(twb.shape[1])]
        emd = earthmover_distance(swbe, twbe)
        bert_earthmovers_feats.append(emd)
        #print("Finished with earthmovers")
        #print(time.time() - start)
      
        # cosine between the CLS tokens
        swb_cls = swb[0][0].reshape(1,-1)
        twb_cls = twb[0][0].reshape(1,-1)
        bert_cls_feats.append(cosine_similarity(swb_cls, twb_cls)[0][0])
        #print("Finished with CLS cosine")
        #print(time.time() - start)
      

      features.append(bert_avg_feats)
      features.append(bert_pwmax_feats)
      features.append(bert_earthmovers_feats)
      features.append(bert_cls_feats)
 
    if self.args.use_pretrained_bpe == 1:
      # BERT feats
      bert_avg_feats = []
      bert_pwmax_feats = []
      bert_earthmovers_feats = []
      
      print("Calculating pretrained bpe  feats ...")
      for sw, tw in word_pairs:
        start = time.time()
        swb = self.multibpemb.embed(sw).T
        twb = self.multibpemb.embed(tw).T
        #print("Finished with bert")
        #print(time.time() - start)

        # cosine of averaged wordpiece embeddings
        swb_avg = swb.mean(1).reshape(1, -1)
        twb_avg = twb.mean(1).reshape(1, -1)
        bert_avg_feats.append(cosine_similarity(swb_avg, twb_avg)[0][0])
        #print("Finished with avg")
        #print(time.time() - start)

        # pairwise max cosine of wordpiece embeddings
        pairwise_sims = cosine_similarity(swb.T, twb.T)
        bert_pwmax_feats.append(pairwise_sims.max())
        #print("Finished with pairwisemax")
        #print(time.time() - start)
      
        # earthmovers distance 
        swbe = [tuple(swb[:,i].tolist()) for i in range(swb.shape[1])]
        twbe = [tuple(twb[:,i].tolist()) for i in range(twb.shape[1])]
        emd = earthmover_distance(swbe, twbe)
        bert_earthmovers_feats.append(emd)
        #print("Finished with earthmovers")
        #print(time.time() - start)

      features.append(bert_avg_feats)
      features.append(bert_pwmax_feats)
      features.append(bert_earthmovers_feats)
 
  
    if self.args.use_frequencies == 1:
      #print("Calculating frequency features ...")
      # word frequencies and word rankings in frequency sorted lists
      src_freq_feats = []
      tar_freq_feats = []
      src_frank_feats = []
      tar_frank_feats = []
      for sw, tw in word_pairs:
        src_frank_feats.append(float(self.src_w2ind[sw]) / len(self.src_w2ind))
        tar_frank_feats.append(float(self.tar_w2ind[tw]) / len(self.tar_w2ind))
      features.append(src_frank_feats)
      features.append(tar_frank_feats)

    return(np.array(features).T)


class PoolerCNG:

  def __init__(self, cache_filename, src_words, tar_words, src_code, tar_code,resort_func = None):
    with(open(cache_filename, "rb")) as infile:
      self.orth_candidates = pickle.load(infile)
    if resort_func is not None:
      self.orth_candidates = sorted(self.orth_candidates, key = lambda t:resort_func(t[0],t[1]), reverse = True)
    # some data needed for frequency filtering later
    FREQ_PERCENTILE = 0.05
    self.src_threshold =  np.percentile([word_frequency(w, src_code) for w in src_words], FREQ_PERCENTILE)
    self.tar_threshold = np.percentile([word_frequency(w, tar_code) for w in tar_words], FREQ_PERCENTILE)
    self.src_code = src_code
    self.tar_code = tar_code

  def generate_candidates(self, current_word_pairs, N, x, z): # x and z are not used :P
    # the list was already sorted during precomputing so no sorting here
    current_word_pairs_hash = set([w1+w2 for w1,w2 in current_word_pairs])
    candidates_filtered = [t for t in self.orth_candidates if t[0]+t[1] not in current_word_pairs_hash]
 
    # length filtering, i was not happy with the candidates obtained in this way so i moved to frequency filtering
    #candidates_filtered = [t for t in candidates_filtered if len(t[0]) < 10 and len(t[1]) < 10]
    #candidates_filtered = [t for t in candidates_filtered if word_frequency(t[0], self.src_code) > self.src_threshold and word_frequency(t[1], self.tar_code) > self.tar_threshold]

    # TOP 0.05% by orthographic similarity then sort those by word freq
    candidates_filtered = candidates_filtered[0:int(0.05 * len(candidates_filtered))]
    candidates_filtered = sorted(candidates_filtered, key = lambda t: word_frequency(t[0], self.src_code) + word_frequency(t[1], self.tar_code), reverse = True)
    candidates_filtered = [t for t in candidates_filtered if t[0].isalpha() and t[1].isalpha()]
    return(candidates_filtered[0:N])


class PoolerMNN:
  def __init__(self, src_i2w, tar_i2w, src_w2i, tar_w2i, src_code, tar_code):
    self.src_ind2w = src_i2w
    self.tar_ind2w = tar_i2w
    self.src_word2ind = src_w2i
    self.tar_word2ind = tar_w2i
    self.src_code = src_code
    self.tar_code = tar_code
 
  def generate_candidates(self, current_word_pairs, N, src_vectors, tar_vectors):
    current_word_pairs_hash = set([w1+w2 for w1,w2 in current_word_pairs])
    print("Calculating SRC - TAR NNs ...")
    nn_s2t = get_1NNfast(list(range(len(self.src_ind2w))), src_vectors, tar_vectors, cuda = True, batch_size = 500, return_scores = True)
    print("Calculating TAR - SRC NNs ...")
    nn_t2s = get_1NNfast(list(range(len(self.tar_ind2w))), tar_vectors, src_vectors, cuda = True, batch_size = 500, return_scores = True)
    print("Sorting and recomputing cosines ...")
    #word index is the key in the dicts and [0],[1] of the value tuple are the index of the nn and the score
    candidates = [(sw,nn_s2t[sw][0],nn_s2t[sw][1]) for sw in nn_s2t if nn_t2s[nn_s2t[sw][0]][0] == sw] # the mutual nearest neighbours 

    candidates_str = [(self.src_ind2w[k], self.tar_ind2w[v], c) for k,v,c in candidates] # turn indexes into words
    candidates_str = [t for t in candidates_str if t[0].isalpha() and t[1].isalpha()]
    candidates_sorted = sorted(candidates_str, key = lambda  t: self.src_word2ind[t[0]] + self.tar_word2ind[t[1]]) # sort by frequency but approximated by ranks in fasttext files
    candidates_filtered = [t for t in candidates_sorted if t[0]+t[1] not in current_word_pairs_hash] # throw out candidates that are already being used
    return(candidates_filtered[0:N])

class PoolerMNNSubsampling:
  def __init__(self, src_i2w, tar_i2w, src_w2i, tar_w2i, src_code, tar_code):
    self.src_ind2w = src_i2w
    self.tar_ind2w = tar_i2w
    self.src_word2ind = src_w2i
    self.tar_word2ind = tar_w2i
    self.src_code = src_code
    self.tar_code = tar_code
    self.num_iter = 10
    self.threshold = 9
    self.subsample_percent = 0.9
    self.pool_multiplier = 5

  def generate_candidates(self, current_word_pairs, N, orig_src_vectors, orig_tar_vectors):
    joint_dict = {}
    for iter in range(self.num_iter):
      # generate subsample aligned embeddings of size, say, 2*N
      current_word_pairs_sample = list(random.sample(current_word_pairs, int(self.subsample_percent * len(current_word_pairs))))
      src_indices = [self.src_word2ind[t[0]] for t in current_word_pairs_sample]
      trg_indices = [self.tar_word2ind[t[1]] for t in current_word_pairs_sample]
      src_words = list(self.src_word2ind.keys())
      trg_words = list(self.tar_word2ind.keys())
      print("Starting the Artetxe et al. alignment ...") 
      src_vectors, tar_vectors = run_supervised_alignment(src_words, trg_words, orig_src_vectors, orig_tar_vectors, src_indices, trg_indices)    
    
      current_word_pairs_hash = set([w1+w2 for w1,w2 in current_word_pairs]) # we throw out all that are in the train dict not only those that are in the sample
      # generate MNNs from those embeddings
      print("Calculating SRC - TAR NNs ...")
      nn_s2t = get_1NNfast(list(range(len(self.src_ind2w))), src_vectors, tar_vectors, cuda = True, batch_size = 500, return_scores = True)
      print("Calculating TAR - SRC NNs ...")
      nn_t2s = get_1NNfast(list(range(len(self.tar_ind2w))), tar_vectors, src_vectors, cuda = True, batch_size = 500, return_scores = True)
      print("Sorting and recomputing cosines ...")
      #word index is the key in the dicts and [0],[1] of the value tuple are the index of the nn and the score
      candidates = [(sw,nn_s2t[sw][0],nn_s2t[sw][1]) for sw in nn_s2t if nn_t2s[nn_s2t[sw][0]][0] == sw] # the mutual nearest neighbours 
      candidates_str = [(self.src_ind2w[k], self.tar_ind2w[v], c) for k,v,c in candidates] # turn indexes into words
      candidates_str = [t for t in candidates_str if t[0].isalpha() and t[1].isalpha()]
      candidates_sorted = sorted(candidates_str, key = lambda  t: self.src_word2ind[t[0]] + self.tar_word2ind[t[1]]) # sort by frequency but approximated by ranks in fasttext files
      candidates_filtered = [t for t in candidates_sorted if t[0]+t[1] not in current_word_pairs_hash] # throw out candidates that are already being used
      
      # generate hash with how many times they appeared in each list
      for sword, tword, score in candidates_filtered[0:self.pool_multiplier * N]:
        c = (sword, tword)
        joint_dict[c] = 1 if c not in joint_dict else joint_dict[c] + 1
  
    # discard all that are below the threshold 
    ret_list = []
    for c in joint_dict:
      if joint_dict[c] >= self.threshold:
        ret_list.append(c)
    # sort the rest by sum of word indices in the fasttext files
    ret_list_sorted = sorted(ret_list, key = lambda  t: self.src_word2ind[t[0]] + self.tar_word2ind[t[1]]) # sort by frequency but approximated by ranks in fasttext files
    print(ret_list_sorted)
    print(len(ret_list_sorted))
    # return first N from the list (if less than N made it to the final list then error or warning)
    return(ret_list_sorted[0:N])




class PoolerCombined:
  def __init__(self, pooler_list): # pooler list has tuples with 1) a pooler object and 2) the percentage of items that will be pooled from that pooler (number 0-1)
    self.poolers = pooler_list
  
  def generate_candidates(self, current_word_pairs, N, x, z): 
    all_candidates = []
    for pooler, p in self.poolers:
      candidates = pooler.generate_candidates(current_word_pairs, int(p*N), x, z)
      all_candidates += candidates
    return(all_candidates)



class Classifier:
  def __init__(self, feat_calc, sc):
    self.fc = feat_calc  
    self.scaler = StandardScaler()
    self.scoring = sc
  
  def fit(self, pos_examples, neg_examples):
    train_set = pos_examples + neg_examples
    labs = [1]*len(pos_examples) + [-1]*len(neg_examples)
    
    print("Calculating features ...")
    feats = self.fc.calc_features(train_set, labs, mode = "train")
    
    feats = self.scaler.fit_transform(feats)

    feats, labs = shuffle(feats, labs)
    feats, labs = np.array(feats), np.array(labs) 
 
    print("Crossvalidation ...")
    best_score, best_h, best_alpha =  -1000, None, None
    for h in [3,5,10,20]:
      for alpha in [0.0001,  0.01, 1]:
        current_model = MLPClassifier(hidden_layer_sizes = (h,), early_stopping = True, alpha = alpha)
        s = cross_val_score(estimator = current_model, X = feats, y = labs, cv = 3, scoring = self.scoring, n_jobs = -1)       
        s = np.mean(s)
        if s > best_score:
          best_score, best_h, best_alpha = s, h, alpha
    self.model = MLPClassifier(hidden_layer_sizes = (best_h,), early_stopping = True, alpha = best_alpha)
    self.model.fit(feats, labs)

    print("Crossval score of best model was " + str(best_score))

  def predict(self, examples):
    feats = self.fc.calc_features(examples, None, mode = "test") 
    feats = self.scaler.transform(feats)
    correct_col = 0 if self.model.classes_[0] == 1 else 1
    retval = list(self.model.predict_log_proba(feats)[:,correct_col])
    return(retval)



# method can be "random", "hard" or "mix"
# hard examples are wrong pairs that (in spite of being wrong) have high cosine
# mix is half random half hard 
def generate_negative_examples(positive_examples, src_w2ind, tar_w2ind, x, z, method = "random"): 
  l = len(positive_examples)
  if method == "mix":
    num_rand, num_hard = math.floor(l / 2), math.ceil(l / 2) # floor/ceil so the sum is still == l
  elif method == "random":
    num_rand, num_hard = l, 0
  elif method == "hard":
    num_rand, num_hard = 0, l
  else:
    raise Exception("Unsupported method for negative sampling.")

  rand_list, hard_list = [], []
  positive_src = [t[0] for t in positive_examples]
  positive_tar = [t[1] for t in positive_examples]

  # generate the random examples (num_rand of them)
  for i in range(num_rand):
    success = False
    while(not success):
      src_ind, tar_ind = randrange(l), randrange(l)
      if src_ind != tar_ind: # when the indexes are the same that is a positive example
        rand_list.append((positive_src[src_ind], positive_tar[tar_ind]))
        success = True
   
  # generate the hard examples (num_hard of them, or skip it if we dont need  them) 
    
  if num_hard > 0:
    pos_src_word_indexes = [src_w2ind[i] for i in positive_src]
    pos_tar_word_indexes = [tar_w2ind[i] for i in positive_tar]
    pos_src_embeddings = x[pos_src_word_indexes,:]
    pos_tar_embeddings = z[pos_tar_word_indexes,:]
    print("Starting dot prod")
    similarities = pos_src_embeddings.dot(pos_tar_embeddings.T)
    print("Finished dot prod")
    l = len(positive_src)

    sflat = similarities.flatten()
    sind = (-sflat).argsort()
     
    current = 0
    for i in range(num_hard): # stupid but works fast enough, do n_hard argmaxes on the array (argmax is very fast and there will only ever be a few thousand of them needed)
      success = False
      while not success:
        ind = sind[current].item()
        current += 1
        si, ti = int(ind / l), ind % l
        if si != ti:
          hard_list.append((positive_src[si], positive_tar[ti]))
          success = True
  
  ret_list = rand_list + hard_list
  shuffle(ret_list)  
  return(ret_list)
 
class OrthoNNProvider():
  def __init__(self, path_to_file, src_w2ind, tar_w2ind, src_ind2w, tar_ind2w):
    with(open(path_to_file, "rb")) as infile:
      self.d = pickle.load(infile)
    self.src_ind2w = src_ind2w
    self.tar_ind2w = tar_ind2w
    self.src_w2ind = src_w2ind
    self.tar_w2ind = tar_w2ind
 
  def get_top_neighbours(self, word, k):
    similar_targets_ind = [t[0] for t in self.d[self.src_w2ind[word]]]
    similar_targets = [self.tar_ind2w[x] for x in similar_targets_ind]
    sim_words = [(w,textdistance.levenshtein.distance(w, word)) for w in similar_targets]
    sorted_sims = sorted(sim_words, key = lambda t:t[1]) # smallest distances first
    return sorted_sims[0:k]
 

# method can be "random", "hard" or "mix"
def generate_negative_examples_v2(positive_examples, src_w2ind, tar_w2ind, src_ind2w, tar_ind2w, x, z, top_k, num_neg_per_pos, num_neg_editdist, ornn): 
  l = len(positive_examples)
  positive_src = [t[0] for t in positive_examples]
  positive_tar = [t[1] for t in positive_examples]

  
  pos_src_word_indexes = [src_w2ind[i] for i in positive_src]
  pos_tar_word_indexes = [tar_w2ind[i] for i in positive_tar]
  
  correct_mapping = dict(zip(pos_src_word_indexes, pos_tar_word_indexes))

  # nns maps src word indexes to a list of tuples (tar_index, similarity)
  nns = get_NN(pos_src_word_indexes, x, z, top_k, cuda = True, mode = "dense", batch_size = 100, return_scores = True, fast1NNmode = False)

  neg_examples = []  
  for src_ind in nns:
    for i in range(num_neg_per_pos):
      sampling_success = False
      while not sampling_success:
        rand_neighbour_ind = nns[src_ind][randrange(top_k)][0]
        if rand_neighbour_ind != correct_mapping[src_ind]:
          sampling_success = True
          neg_examples.append((src_ind2w[src_ind], tar_ind2w[rand_neighbour_ind]))

  
  neg_examples_ed = []
  if num_neg_editdist > 0:
    for src_w in tqdm(positive_src):
      top_ortho_nns = ornn.get_top_neighbours(src_w, top_k)
      for i in range(num_neg_editdist):
         sampling_success = False
         while not sampling_success:
          rand_neighbour = top_ortho_nns[randrange(top_k)][0]
          if tar_w2ind[rand_neighbour] != correct_mapping[src_w2ind[src_w]]:
            sampling_success = True
            neg_examples_ed.append((src_w, rand_neighbour))
  return_list = neg_examples + neg_examples_ed
  shuffle(return_list) 
  return return_list
 


def run_selflearning(args):
  SL_start_time = time.time()
  SELF_LEARNING_ITERATIONS = args.num_iterations
  EXAMPLES_TO_POOL = args.examples_to_pool
  EXAMPLES_TO_ADD = args.examples_to_add
  #neg_method = "hard"
  neg_top_k = 10
  neg_per_pos = 1
  neg_editdist_per_pos = 1

  write_original = False
  use_classifier = args.use_classifier == 1  
  #add_cng_pooler = False
  use_mnns_pooler = args.use_mnns_pooler == 1
 
 
  task_name = args.src_lid + "-" + args.tar_lid + "-" + args.idstring
  cng_cachefile = "./cache/ortho_nn_" + task_name + ".pickle"
  cng_nn_file = "./cache/ortho_nn_" + task_name + "-dict.pickle"
  model_out_filename = args.model_filename
 

  # load up the embeddings
  print("Loading embeddings from disk ...")
  dtype = "float32"
  srcfile = open(args.in_src, encoding="utf-8", errors='surrogateescape')
  trgfile = open(args.in_tar, encoding="utf-8", errors='surrogateescape')
  src_words, x = embeddings.read(srcfile, dtype=dtype)
  trg_words, z = embeddings.read(trgfile, dtype=dtype)

  # load the supervised dictionary
  src_word2ind = {word: i for i, word in enumerate(src_words)}
  trg_word2ind = {word: i for i, word in enumerate(trg_words)}
  src_ind2word = {i: word for i, word in enumerate(src_words)}
  trg_ind2word = {i: word for i, word in enumerate(trg_words)}

  # precomputing of orthographic nns, if needed
  if not os.path.isfile(cng_cachefile) or not os.path.isfile(cng_nn_file):
    print("Precomputing ortographic neighbours (will be cached for further calls)")
    neighbours, neighbours_dict = precompute_orthographic_NN(src_words, trg_words, 50, cuda = True)

    print("Writing top pairs to file ...")
    with open(cng_cachefile, "wb") as outfile: 
      pickle.dump(neighbours, outfile)

    print("Writing precomputed ortho nns to file ...")
    with open(cng_nn_file, "wb") as outfile: 
      pickle.dump(neighbours_dict, outfile)

  #if neg_editdist_per_pos > 0:
  print("Loading the precomputed orthography nns ...")
  or_nn_provider = OrthoNNProvider(cng_nn_file, src_word2ind, trg_word2ind, src_ind2word, trg_ind2word)
 
  src_indices, trg_indices, pos_examples = [], [], []
  f = open(args.train_dict, encoding="utf-8", errors='surrogateescape')
  for line in f:
      src, trg = [x.lower().strip() for x in line.split("\t")]
      pos_examples.append((src,trg))

  src_indices = [src_word2ind[t[0]] for t in pos_examples]
  trg_indices = [trg_word2ind[t[1]] for t in pos_examples]

  # call artetxe to get the initial alignment on the initial train dict
  print("Starting the Artetxe et al. alignment ...") 
  xw, zw = run_supervised_alignment(src_words, trg_words, x, z, src_indices, trg_indices, supervision = args.art_supervision)    
   
  if write_original:
    src_output_filename = "./SRC_SUPERVISED_" + task_name + "-nosl.txt"
    tar_output_filename = "./TAR_SUPERVISED_" + task_name + "-nosl.txt"
    srcfile = open(src_output_filename, mode='w', encoding="utf-8", errors='surrogateescape')
    trgfile = open(tar_output_filename, mode='w', encoding="utf-8", errors='surrogateescape')
    embeddings.write(src_words, xw, srcfile)
    embeddings.write(trg_words, zw, trgfile)
    srcfile.close()
    trgfile.close()
 
  # generate negative examples for the current 
  print("Generating negative examples ...")
  #neg_examples = generate_negative_examples(pos_examples, src_word2ind, trg_word2ind, xw, zw, method = neg_method)
  neg_examples =  generate_negative_examples_v2(pos_examples, src_word2ind, trg_word2ind, src_ind2word, trg_ind2word, xw, zw, top_k = neg_top_k, num_neg_per_pos = neg_per_pos, num_neg_editdist = neg_editdist_per_pos, ornn = or_nn_provider) 


  if use_classifier: 
    print("Training initial classifier ...")
    #train initial classifier on the seed dictionary
    feat_calculator = FeatureCalculator(src_words, trg_words, xw, zw, args.src_lid, args.tar_lid, or_nn_provider, args)
    model = Classifier(feat_calculator, args.scoring)
    model.fit(pos_examples, neg_examples)

  #print(model.predict([("dog","pas"),("cat","sabor")]))
  #exit()
  pooler_mnn = PoolerMNN(src_ind2word, trg_ind2word, src_word2ind, trg_word2ind, args.src_lid, args.tar_lid)
  pooler_mnns = PoolerMNNSubsampling(src_ind2word, trg_ind2word, src_word2ind, trg_word2ind, args.src_lid, args.tar_lid)

  
  if use_mnns_pooler:
    pooler = pooler_mnns
  else:
    pooler = pooler_mnn

  print("Starting self-learning iterations")
  for it in [a+1 for a in range(SELF_LEARNING_ITERATIONS)]:
    print(" *************** ITERATION %d ****************** " % (it))
    #print("Pooling for extra alignment candidates ...")
    # call pooler to get extra aligned candidates
    if not use_mnns_pooler: # regular MNN is done on the aligned matrices
      pooled_examples = pooler.generate_candidates(pos_examples, EXAMPLES_TO_POOL, xw, zw)
    else: # MNNS is done on original matrices (because it runs the alignment internally multiple times)
      pooled_examples = pooler.generate_candidates(pos_examples, EXAMPLES_TO_POOL, x, z)

    pooled_examples = [(t[0],t[1]) for t in pooled_examples] # the other classes expect tuples of 2 not 3 (without the scores)
 
    if len(pooled_examples) == 0:
      print("All MNN that are induced by the model are already in the training list")
      break

    if use_classifier:
      print("Predicting scores for the candidates ...")
      #predict scores for the  candidates using the classifier
      scores = model.predict(pooled_examples)    
      assert len(scores) == len(pooled_examples)

      scored_examples = [(pooled_examples[i], scores[i]) for i in range(len(pooled_examples))]
      top_examples = sorted(scored_examples, key = lambda t:t[1], reverse = True)[0:EXAMPLES_TO_ADD]
      top_examples = [t[0] for t in top_examples] # removes the scores
    else:  
      # **** test bypassing the classifier ***** just adding MNN pooled examples
      top_examples = pooled_examples[0:EXAMPLES_TO_ADD] 
   
    print("Added top " + str(EXAMPLES_TO_ADD) + " examples to the train dictionary.")
    #print(top_examples[0:10])
    #print("Before: " + str(len(pos_examples)))
    # add top N to the expanded seed dictionary
    pos_examples += top_examples 
    #print("After: " + str(len(pos_examples)))

    print("Rerunning Artetxe with the updated train dictionary ...")
    # rerun the alignment algorithm with the updated dict to get updated x and z
    src_indices = [src_word2ind[t[0]] for t in pos_examples]
    trg_indices = [trg_word2ind[t[1]] for t in pos_examples]
    print(len(src_indices))
    print(len(trg_indices))
    new_xw, new_zw = run_supervised_alignment(src_words, trg_words, x, z, src_indices, trg_indices, supervision = args.art_supervision)    
    xw, zw = new_xw, new_zw
    if use_classifier:
      feat_calculator.update_embeddings(xw,zw) # will use the new aligned space when calculating features next time 
 
    print("Generating negative samples for the expanded train dictionary.")
    # generate new negative examples (these will be different in every iter, maybe change this?)
    #neg_examples = generate_negative_examples(pos_examples, src_word2ind, trg_word2ind, xw, zw, method = neg_method)
    neg_examples =  generate_negative_examples_v2(pos_examples, src_word2ind, trg_word2ind, src_ind2word, trg_ind2word, xw, zw, top_k = neg_top_k, num_neg_per_pos = neg_per_pos, num_neg_editdist = neg_editdist_per_pos, ornn = or_nn_provider) 
    print("Retraining the classifier on the expanded data ...")
    # retrain the classifier on the expanded seed dictionary and the new (theoretically better) alignments

    if use_classifier:
      model.fit(pos_examples, neg_examples)
    
    #if it in [1,2,3,5,10,20,30,40,50]:
    if args.checkpoint_steps != -1 and it % args.checkpoint_steps == 0:
      print("Writing output to files ...")
      # write res to disk
      srcfile = open(args.out_src + ".checkpoint-" + str(it), mode='w', encoding="utf-8", errors='surrogateescape')
      trgfile = open(args.out_tar + ".checkpoint-" + str(it), mode='w', encoding="utf-8", errors='surrogateescape')
      embeddings.write(src_words, xw, srcfile)
      embeddings.write(trg_words, zw, trgfile)
      srcfile.close()
      trgfile.close()
      if use_classifier:
        print("Saving the supervised model to disk ...")
        with open(model_out_filename + ".checkpoint-" + str(it), "wb") as outfile:
          pickle.dump(model, outfile)
  
  
  if SELF_LEARNING_ITERATIONS == 0:
    it = 0

  print("Writing output to files ...")
  # write res to disk
  srcfile = open(args.out_src, mode='w', encoding="utf-8", errors='surrogateescape')
  trgfile = open(args.out_tar, mode='w', encoding="utf-8", errors='surrogateescape')
  embeddings.write(src_words, xw, srcfile)
  embeddings.write(trg_words, zw, trgfile)
  srcfile.close()
  trgfile.close()
  if use_classifier:
    print("Saving the supervised model to disk ...")
    with open("./" + model_out_filename, "wb") as outfile:
      pickle.dump(model, outfile)
  print(str(args.idstring))
  print("SL FINISHED " + str(time.time() - SL_start_time))
  

if __name__ == "__main__":  

   parser = argparse.ArgumentParser(description='Run classification based self learning for aligning embedding spaces in two languages.')

   parser.add_argument('--train_dict', type=str, help='Name of the input dictionary file.', required = True)
   parser.add_argument('--in_src', type=str, help='Name of the input source languge embeddings file.', required = True)
   parser.add_argument('--in_tar', type=str, help='Name of the input target language embeddings file.', required = True)
   parser.add_argument('--out_src', type=str, help='Name of the output source languge embeddings file.', required = True)
   parser.add_argument('--out_tar', type=str, help='Name of the output target language embeddings file.', required = True)
   parser.add_argument('--src_lid', type=str, help='Source language id.', required = True)
   parser.add_argument('--tar_lid', type=str, help='Target language id.', required = True)
   parser.add_argument('--model_filename', type=str, help='Name of file where the model will be stored..', required = True)
   parser.add_argument('--idstring', type=str,  default="EXP", help='Special id string that will be included in all generated model and cache files. Default is EXP.')

   parser.add_argument('--scoring', type=str,  default='f1_macro', help='Scoring type for the classifier, can be any string valid in sklearn. Default is f1_macro.')
   parser.add_argument('--num_iterations', type=int,  default=10, help='Number of self learning iterations to run. Default is 10.')
   parser.add_argument('--examples_to_pool', type=int,  default=5000, help='Number of examples to pool in each self learning iteration. Default is 5000.')
   parser.add_argument('--examples_to_add', type=int,  default=500, help='Number of examples from the pool to add to the train set in each self learning iteration. Default is 500.')
   parser.add_argument('--use_mnns_pooler', type=int,  default=0, help='Whether to MNN stochastic pooler instead of regular MNN pooler (1 for yes 0 for no). Default is 0.')
   parser.add_argument('--use_classifier', type=int,  default=1, help='Whether to use the classifier to rerank pooled candidates. Default is 1.')

   parser.add_argument('--use_edit_dist', type=int,  default=1, help='Whether to use edit distance features (1 for yes 0 for no). Default is 1.')
   parser.add_argument('--use_aligned_cosine', type=int,  default=1, help='Whether to use cosing distance in aligned space features (1 for yes 0 for no). Default is 1.')
   parser.add_argument('--use_ngrams', type=int,  default=1, help='Whether to use ngram overlap features (1 for yes 0 for no). Default is 1.')
   parser.add_argument('--use_full_bert', type=int,  default=0, help='Whether to use bert based features (1 for yes 0 for no). Default is 0.')
   parser.add_argument('--use_pretrained_bpe', type=int,  default=0, help='Whether to use pretrained BPE features (1 for yes 0 for no). Default is 0.')
   parser.add_argument('--use_frequencies', type=int,  default=1, help='Whether to use frequency features (1 for yes 0 for no). Default is 1.')
   parser.add_argument('--use_aligned_pca', type=int,  default=1, help='Whether to use PCA reduced embeddings in the aligned space as features (1 for yes 0 for no). Default is 1.')
   parser.add_argument('--use_char_ngrams', type=int,  default=0, help='Whether to use character ngrams as features (1 for yes 0 for no). Default is 0.')

   parser.add_argument('--art_supervision', type=str,  default="--supervised", help='Supervision argument to pass on to Artetxe et al. code. Default is "--supervised".')
   parser.add_argument('--checkpoint_steps', type=int,  default=-1, help='A checkpoint will be saved every checkpoint_steps iterations. -1 to skip saving checkpoints. Default is -1.')


   args = parser.parse_args()

   run_selflearning(args)

