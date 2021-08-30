# ClassyMap
Tool for aligning word vectors and inducing bilingual lexicons.

Code and data of the following paper:  [Classification-Based Self-Learning for Weakly Supervised Bilingual Lexicon Induction](https://aclanthology.org/2020.acl-main.618.pdf)

If you use this code please cite the paper:

```
@inproceedings{karan2020classification,
  title={Classification-based self-learning for weakly supervised bilingual lexicon induction},
  author={Karan, Mladen and Vuli{\'c}, Ivan and Korhonen, Anna and Glava{\v{s}}, Goran},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  pages={6915--6922},
  year={2020}
}
```

We also recommend you cite the original VecMap paper that this code heavily relies on.

```
@inproceedings{artetxe2018acl,
  author    = {Artetxe, Mikel  and  Labaka, Gorka  and  Agirre, Eneko},
  title     = {A robust self-learning method for fully unsupervised cross-lingual mappings of word embeddings},
  booktitle = {Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year      = {2018},
  pages     = {789--798}
}
```

### Quickstart:
##### NOTE: after checking out the repo, we recommend using a python virtual environment, testing was done using anaconda:
```
conda create -n "cm-env" python=3.6.9
conda activate "cm-env"
pip install -r src/requirements.txt
```
##### NOTE: the following steps can be applied to any language pair, as a working example we will use English and Croatian.

##### NOTE: we recommend running steps 2 and 3 with cuda, without it they are both somewhat slower and not well tested

### 1. Get the embedding files

Run the ```./embeddings/dl-embeddings-all.sh``` script, to download only embeddings used in the example below (en and hr). Alternatively, run the ```./embeddings/dl-embeddings-all.sh``` script, to download all embeddings used in the paper

#### If you need to work on a different language pair follow the instructions below.
--------------------

Download the embedding matrices (we used those from https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/ in text format). For example:

English → https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec

Croatian → https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.hr.vec

Transform each matrix file to leave only the top 200 000 most frequent words for each (this should also be reflected in the counts in the first line of the matrix file). 
Provided the words in matrices are sorted descending by frequency in the files, you can do this with:  

```
head -n 200000 wiki.hr.vec | sed "1s/.*/199999 300/" > wiki.200k.hr.vec 
head -n 200000 wiki.en.vec | sed "1s/.*/199999 300/" > wiki.200k.en.vec 
```

##### NOTE: This filtering step is not mandatory but it should speed things up without significant impact on quality.
##### NOTE: if your source and target languages use different scripts you must transliterate words in both the embedding matrices and the train/test files to use the same script (most likely latin) for both languages  before running subsequent steps. Scripts "translit-matrix.py" and "translit-dict.py" for this are included in the src folder, use them as:  
```
python translit-matrix.py src_matrix.vec src_lang_code output_matrix.vec
python translit-dict.py src_dict.tsv src_lang_code trg_lang_code output_dict.tsv WHICH   (WHICH can be "SRC", "TRG" or "BOTH" depending on which parts of the dictionary need transliteration)  
```
for example

```
python translit-matrix.py wiki.200k.ru.vec ru wiki.200k.ru.latin.vec
python translit-dict.py ../data/en-ru/yacle.train.freq.500.en-ru.tsv en ru ../data/en-ru/yacle.train.freq.500.en-ru.transliterated.tsv TRG 	
```

##### NOTE: It is based on the https://pypi.org/project/transliterate/ library and will work for most cryllic scripts and greek, other scripts are currently not supported.

### 2. Running the semi-supervised model 
```
python classymap.py --train_dict trainfile.tsv --in_src in_src_embeddings.vec --in_tar in_tar_embeddings.vec --src_lid "en" --tar_lid "hr" --out_src out_src_embeddings.vec --out_tar out_tar_embeddings.vec --model_filename modefile.pickle --idstring experiment_id
```

for example

```
python classymap.py --train_dict "../data/en-hr/yacle.train.freq.500.en-hr.tsv" --in_src "../embeddings/wiki.200k.en.vec" --in_tar "../embeddings/wiki.200k.hr.vec" --src_lid "en" --tar_lid "hr" --out_src "../embeddings/wiki.200k.en-hr-aligned.EN.vec" --out_tar "../embeddings/wiki.200k.en-hr-aligned.HR.vec" --model_filename "./models/ENHR-model.pickle" --idstring ENHRFASTTEXT
```

##### Note: type ```python classymap.py --help``` for many more options, but the default values for all of them should give reasonable perfomance.
##### Note: ```idstring``` is an arbitrary string that will be used for caching. To avoid caching problems, we recommend using the same ```idstring``` for experiments using the same pair of initial embedding matrices and (more importantly) different ```idstring``` for experiments using different pairs of initial embedding matrices

### 3. Generating and evaluating translation candidates 

#### 3.1. Without reranking:    

```
python eval.py src.aligned.embeddings.vec trg.aligned.embeddings.vec -d dictionary.tsv --cuda 
```

for example:

```
python eval.py "../embeddings/wiki.200k.en-hr-aligned.EN.vec" "../embeddings/wiki.200k.en-hr-aligned.HR.vec" -d ../data/en-hr/yacle.test.freq.2k.en-hr.tsv --cuda
```

#### 3.2. With reranking

add  ```"--super --model model_path --src_lid sl --tar_lid tl --idstring id"``` to all the above command

```model_path``` is the model file you generated in the step 2

```src_lid```, ```tar_lid```, and ```idstring``` should be the same you provided in step 2

An example of expanding the command from step 3.1. is as follows:

```
python eval.py "../embeddings/wiki.200k.en-hr-aligned.EN.vec" "../embeddings/wiki.200k.en-hr-aligned.HR.vec" -d ../data/en-hr/yacle.test.freq.2k.en-hr.tsv --cuda  --super --model "./models/ENHR-model.pickle" --src_lid "en" --tar_lid "hr" --idstring ENHRFASTTEXT
```

##### Note: Regardless of using reranking or not. You can add "--output_file filename.txt" to dump the generated candidates into an output file of the following format:
```
src_word_1 -->	first_target_candidate 	second_target_candidate ....
src_word_2 -->	first_target_candidate 	second_target_candidate ....
...
```

The candidates are tab separated. 
You can specifiy ```"--output_top_K K"``` to limit the number of candidates, e.g. ```"--output_top_K 500"``` to limit each line in the file to the top 500 candidates.

