import sys
import embeddings
import pickle
import numpy as np
from transliterate import translit

emb_filename = sys.argv[1]
lang_code = sys.argv[2]
tar_filename = sys.argv[3]

print("Loading the file ...")
emb_file = open(emb_filename, encoding="utf-8", errors='surrogateescape')
src_words, x = embeddings.read(emb_file, dtype = "float32")

print("Transliterating src words:")
trans_src_words = [translit(x, lang_code.lower(), reversed = True) for x in src_words]

print("New vocab sample:")
print(trans_src_words[0:25])

trgfile = open(tar_filename, mode='w', encoding="utf-8", errors='surrogateescape')
embeddings.write(trans_src_words, x, trgfile)

