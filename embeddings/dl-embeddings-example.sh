wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.hr.vec
head -n 200000 wiki.en.vec | sed "1s/.*/199999 300/" > wiki.200k.en.vec 
head -n 200000 wiki.hr.vec | sed "1s/.*/199999 300/" > wiki.200k.hr.vec 
