# Copyright (C) 2016-2018  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import embeddings
from cupy_utils import *

import argparse
import collections
import numpy as np
import sys

import pickle
from final import *
import gc

from transliterate import translit

BATCH_SIZE = 250

def topk_mean(m, k, inplace=False):  # TODO Assuming that axis is 1
    xp = get_array_module(m)
    n = m.shape[0]
    ans = xp.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = xp.array(m)
    ind0 = xp.arange(n)
    ind1 = xp.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        m.argmax(axis=1, out=ind1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    return ans / k


def supervised_reranking(translations, model_filename, si2w, ti2w, sw2i, tw2i, K, correct_trans, thr):
  with(open(model_filename, "rb")) as infile:
    model = pickle.load(infile)
  
  #print(model.predict([("dog","pas")]))
  # exit(0)

  print("Applying classifier")
  num_iter = 0
  ret = dict()
  verbose = True
 
  for sw in translations:
    num_iter += 1  
    print("Generating predictions of source word " + str(num_iter) + " / " + str(len(translations)))
    candidate_tar_words = translations[sw][0:K].tolist()

    
      
    if verbose:
      print("Src word:" + si2w[sw])
      print("Translations before:" + str([ti2w[x] for x in candidate_tar_words]))
    #print(candidate_tar_words)
    data = [(si2w[sw], ti2w[tw]) for tw in candidate_tar_words]
    tw_scores = model.predict(data) 
    #print(tw_scores)
    assert len(candidate_tar_words) == len(tw_scores)
    scored_tw = zip(candidate_tar_words, tw_scores)
    sorted_tw = sorted(scored_tw, key = lambda t:t[1], reverse = True)
    if verbose:
      print("Translations after rerank: " + str([(ti2w[x[0]], x[1]) for x in sorted_tw]))
    #exit()
    reranked_top = [x[0] for x in sorted_tw]
    rest_of_list = translations[sw][K:].tolist()
    ret[sw] = reranked_top + rest_of_list
    if verbose:
      print("Gold acceptable translations:")
      print(str([ti2w[x] for x in correct_trans[sw]]))
      print("*********************")
  return(ret)

def supervised_reranking2(translations, model_filename, si2w, ti2w, sw2i, tw2i, K, correct_trans, thr, ornn, numor):
  with(open(model_filename, "rb")) as infile:
    model = pickle.load(infile)
  
  #print(model.predict([("dog","pas")]))
  # exit(0)

#  print("Applying classifier")
  num_iter = 0
  ret = dict()
  verbose = False
 
  for sw in translations:
    num_iter += 1  
#    print("Generating predictions of source word " + str(num_iter) + " / " + str(len(translations)))
    candidate_tar_words = translations[sw][0:K].tolist()
    if numor > 0:
      #print("Src word:" + si2w[sw])
      
      additional_ortho_candidates = [tw2i[x[0]] for x in ornn.get_top_neighbours(si2w[sw],numor)] # target indices for the ortographically similar candidates
      #print([ti2w[x] for x in additional_ortho_candidates])
      candidate_tar_words += additional_ortho_candidates
      candidate_tar_words_set = set(candidate_tar_words)
      candidate_tar_words = list(candidate_tar_words_set)
    else:
      candidate_tar_words_set = None
     

    if verbose:
      print("Src word:" + si2w[sw])
      print("Translations before:" + str([ti2w[x] for x in candidate_tar_words]))
    #print(candidate_tar_words)
    data = [(si2w[sw], ti2w[tw]) for tw in candidate_tar_words]
    tw_scores = model.predict(data)
    tw_scores = [1.0 / (1.0 + math.exp(-x)) for x in tw_scores] # sigmoid the output of decision function
#    print(tw_scores)
    assert len(candidate_tar_words) == len(tw_scores)
    scored_tw = list(zip(candidate_tar_words, tw_scores))
    
   
    scored_tw_above_thr = [x for x in scored_tw if x[1] >= thr]
    scored_tw_below_thr = [x for x in scored_tw if x[1] < thr]
#    print([x[1]  for x in scored_tw])
#    print(scored_tw_above_thr)
#    print(scored_tw_below_thr)    

    sorted_tw_above_thr = sorted(scored_tw_above_thr, key = lambda t:t[1], reverse = True)
    sorted_tw_below_thr = scored_tw_below_thr # no sorting will be in the same order as in the original list
 #   print(sorted_tw_above_thr)
 #   print(sorted_tw_below_thr)

    if verbose:
      print("Translations after rerank: " + str([(ti2w[x[0]], x[1]) for x in sorted_tw]))
    #exit()
    reranked_top = [x[0] for x in sorted_tw_above_thr] + [x[0] for x in sorted_tw_below_thr] 
#    print(reranked_top)
#    exit()

    rest_of_list = translations[sw][K:].tolist()
    if candidate_tar_words_set is not None: # ensure that any ortographic candidates added tp the top candidates list for reranking are removed from the rest of the list
      rest_of_list = [x for x in rest_of_list if x not in candidate_tar_words_set]
    ret[sw] = reranked_top + rest_of_list
    if verbose:
      print("Gold acceptable translations:")
      print(str([ti2w[x] for x in correct_trans[sw]]))
      print("*********************")
  return(ret)




def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate embeddings of two languages in a shared space in word translation induction')
    parser.add_argument('src_embeddings', help='the source language embeddings')
    parser.add_argument('trg_embeddings', help='the target language embeddings')
    parser.add_argument('-d', '--dictionary', default=sys.stdin.fileno(), help='the test dictionary file (defaults to stdin)')
    parser.add_argument('--retrieval', default='nn', choices=['nn', 'invnn', 'invsoftmax', 'csls'], help='the retrieval method (nn: standard nearest neighbor; invnn: inverted nearest neighbor; invsoftmax: inverted softmax; csls: cross-domain similarity local scaling)')
    parser.add_argument('--inv_temperature', default=1, type=float, help='the inverse temperature (only compatible with inverted softmax)')
    parser.add_argument('--inv_sample', default=None, type=int, help='use a random subset of the source vocabulary for the inverse computations (only compatible with inverted softmax)')
    parser.add_argument('-k', '--neighborhood', default=10, type=int, help='the neighborhood size (only compatible with csls)')
    parser.add_argument('--dot', action='store_true', help='use the dot product in the similarity computations instead of the cosine')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--seed', type=int, default=0, help='the random seed')
    parser.add_argument('--precision', choices=['fp16', 'fp32', 'fp64'], default='fp32', help='the floating-point precision (defaults to fp32)')
    parser.add_argument('--cuda', action='store_true', help='use cuda (requires cupy)')
    parser.add_argument('--super', action='store_true', help='use a supervised model to rerank top candidates')
    parser.add_argument('--model', help = "path to supervised model for reranking", type=str)
    parser.add_argument('--output_file', help = "Path to output file (contains a list of target words generated for each source word), if empty no output file will be generated. Default is empty.", type=str, default="")
    
    parser.add_argument('--src_lid', help = "Source language id", type=str)
    parser.add_argument('--tar_lid', help = "Target language id", type=str)
    parser.add_argument('--idstring', help = "Idstring used to look up things in the cache.", type=str)


    parser.add_argument('--num_ort', help = "Number of ortographica neighbours included in the reranking. Default is 3.", type=int, default = 3)
    parser.add_argument('--K', help = "Number of top neightbours in the aligned vector space included in the reranking. Default is 3.", type=int, default = 3)
    parser.add_argument('--threshold', help = "All reranking candidates that have classifier confidence score below  (<) of the threshold are left in their original order. Default is 0.0 (all candidates are reranked).", type=float, default = 0.0)

    args = parser.parse_args()
    
    # Choose the right dtype for the desired precision
    if args.precision == 'fp16':
        dtype = 'float16'
    elif args.precision == 'fp32':
        dtype = 'float32'
    elif args.precision == 'fp64':
        dtype = 'float64'

    # Read input embeddings
    srcfile = open(args.src_embeddings, encoding=args.encoding, errors='surrogateescape')
    trgfile = open(args.trg_embeddings, encoding=args.encoding, errors='surrogateescape')
    src_words, x = embeddings.read(srcfile, dtype=dtype)
    trg_words, z = embeddings.read(trgfile, dtype=dtype)
    
    # NumPy/CuPy management
    if args.cuda:
        if not supports_cupy():
            print('ERROR: Install CuPy for CUDA support', file=sys.stderr)
            sys.exit(-1)
        xp = get_cupy()
        x = xp.asarray(x)
        z = xp.asarray(z)
    else:
        xp = np
    xp.random.seed(args.seed)

    # Length normalize embeddings so their dot product effectively computes the cosine similarity
    if not args.dot:
        embeddings.length_normalize(x)
        embeddings.length_normalize(z)

    # Build word to index map
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}
    src_ind2word = {i: word for i, word in enumerate(src_words)}
    trg_ind2word = {i: word for i, word in enumerate(trg_words)}


    # Read dictionary and compute coverage
    f = open(args.dictionary, encoding=args.encoding, errors='surrogateescape')
    src2trg = collections.defaultdict(set)
    oov = set()
    vocab = set()
    for line in f:
        src, trg = [x.lower().strip() for x in line.split("\t")]
        try:
            src_ind = src_word2ind[src]
            trg_ind = trg_word2ind[trg]
            src2trg[src_ind].add(trg_ind)
            vocab.add(src)
        except KeyError:
            oov.add(src)
    src = list(src2trg.keys())
    oov -= vocab  # If one of the translation options is in the vocabulary, then the entry is not an oov
    coverage = len(src2trg) / (len(src2trg) + len(oov))

   
    # Find translations
    translation = collections.defaultdict(int)
    if args.retrieval == 'nn':  # Standard nearest neighbor
        for i in range(0, len(src), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(src))
            similarities = x[src[i:j]].dot(z.T)
            nn = (-similarities).argsort(axis=1)
            for k in range(j-i):                
                translation[src[i+k]] = nn[k]
    elif args.retrieval == 'invnn':  # Inverted nearest neighbor
        best_rank = np.full(len(src), x.shape[0], dtype=int)
        best_sim = np.full(len(src), -100, dtype=dtype)
        for i in range(0, z.shape[0], BATCH_SIZE):
            j = min(i + BATCH_SIZE, z.shape[0])
            similarities = z[i:j].dot(x.T)
            ind = (-similarities).argsort(axis=1)
            ranks = asnumpy(ind.argsort(axis=1)[:, src])
            sims = asnumpy(similarities[:, src])
            for k in range(i, j):
                for l in range(len(src)):
                    rank = ranks[k-i, l]
                    sim = sims[k-i, l]
                    if rank < best_rank[l] or (rank == best_rank[l] and sim > best_sim[l]):
                        best_rank[l] = rank
                        best_sim[l] = sim
                        translation[src[l]] = k
    elif args.retrieval == 'invsoftmax':  # Inverted softmax
        sample = xp.arange(x.shape[0]) if args.inv_sample is None else xp.random.randint(0, x.shape[0], args.inv_sample)
        partition = xp.zeros(z.shape[0])
        for i in range(0, len(sample), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(sample))
            partition += xp.exp(args.inv_temperature*z.dot(x[sample[i:j]].T)).sum(axis=1)
        for i in range(0, len(src), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(src))
            p = xp.exp(args.inv_temperature*x[src[i:j]].dot(z.T)) / partition
            nn = p.argmax(axis=1).tolist()
            for k in range(j-i):
                translation[src[i+k]] = nn[k]
    elif args.retrieval == 'csls':  # Cross-domain similarity local scaling
        knn_sim_bwd = xp.zeros(z.shape[0])
        for i in range(0, z.shape[0], BATCH_SIZE):
            j = min(i + BATCH_SIZE, z.shape[0])
            knn_sim_bwd[i:j] = topk_mean(z[i:j].dot(x.T), k=args.neighborhood, inplace=True)
        for i in range(0, len(src), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(src))
            similarities = 2*x[src[i:j]].dot(z.T) - knn_sim_bwd  # Equivalent to the real CSLS scores for NN
            nn = (-similarities).argsort(axis=1)
            for k in range(j-i):
                translation[src[i+k]] = nn[k]
    
    #print(translation[src[0]])
    #print(np.where(translation[src[0]] == 15))
    #print(np.where(translation[src[0]] == 15)[0][0])
    
    
    # apply supervised if needed
    if args.super:
      print("Loading model ...")
      model_filename = args.model
      print("Loading ortography stuff ...")
      ortpath = "./cache/ortho_nn_" + args.src_lid + "-" + args.tar_lid + "-" + args.idstring + "-dict.pickle"
      ornn = OrthoNNProvider(ortpath, src_word2ind, trg_word2ind, src_ind2word, trg_ind2word)

      translation_tmp = supervised_reranking2(translation, model_filename, src_ind2word, trg_ind2word, src_word2ind, trg_word2ind, rgs.K, src2trg, args.threshold, ornn, args.num_ort)
      positions = [np.min([translation_tmp[i].index(x) + 1 for x in src2trg[i]]) for i in src] 
      translation = translation_tmp
      #p1 = len([p for p in positions if p == 1]) / len(positions)
      #p5 = len([p for p in positions if p <= 5]) / len(positions)
      #print("K = %d, T = %.3f, P1 = %.3f, P5 = %.3f" % (K, threshold, p1, p5))

      #exit()
    
    else:
      # src2trg[i] are gold translations, look them up in trnaslation[i] and pick the top ranked (min index) one
      positions = [np.min([np.where(translation[i] == x)[0][0]+1 for x in src2trg[i]]) for i in src] 

    assert len(positions) == len(src)

    p1 = len([p for p in positions if p == 1]) / len(positions)
    p5 = len([p for p in positions if p <= 5]) / len(positions)
    p10 = len([p for p in positions if p <= 10]) / len(positions)
    mrr = sum([1.0/p for p in positions]) / len(positions)

    print("P1 = %.4f" % (p1))
    print("P5 = %.4f" % (p5))
    print("P10 = %.4f" % (p10))
    print("MRR = %.4f" % (mrr))

    print('Coverage:{0:7.2%}'.format(coverage))

    if args.output_file.strip() != "":
        print("Writing translation candidates to " + args.output_file)
        with open(args.output_file, "w") as of:
            line = ""
            for sw in translation:
                line += src_ind2word[sw] + " --> "
                for tw in translation[sw].tolist()[0:250]:
                    line += "\t" + trg_ind2word[tw] 
                line += "\n"
            of.write(line)
        

    #print("RAW_OUTPUTS")
    #for p in positions:
      #print(p)
      
 
if __name__ == '__main__':
    main()
