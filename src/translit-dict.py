import sys
import embeddings
import pickle
import numpy as np
from transliterate import translit
import pandas as pd

dict_filename = sys.argv[1]
src_lang_code = sys.argv[2]
tar_lang_code = sys.argv[3]
output_filename = sys.argv[4]
mode = sys.argv[5]

df = pd.read_csv(dict_filename, sep = "\t", header = None)
df.columns = ["src", "trg"]

mode = mode.upper()
if mode != "TRG" and mode != "SRC" and mode != "BOTH":
    raise Exception("Unknown mode: " + mode)

print("Transliterating words ...")
if mode == "SRC" or mode == "BOTH":
   df["src"] = [translit(x, src_lang_code.lower(), reversed = True) for x in df["src"]]

if mode == "TRG" or mode == "BOTH":
   df["trg"] = [translit(x, tar_lang_code.lower(), reversed = True) for x in df["trg"]]

print(df.head())

df.to_csv(output_filename, sep = "\t", index = False, header = False)

