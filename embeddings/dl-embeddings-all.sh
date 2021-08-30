wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.de.vec
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.fi.vec
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.fr.vec
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.hr.vec
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.it.vec
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ru.vec
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.tr.vec

head -n 200000 wiki.de.vec | sed "1s/.*/199999 300/" > wiki.200k.de.vec 
head -n 200000 wiki.en.vec | sed "1s/.*/199999 300/" > wiki.200k.en.vec 
head -n 200000 wiki.fi.vec | sed "1s/.*/199999 300/" > wiki.200k.fi.vec 
head -n 200000 wiki.fr.vec | sed "1s/.*/199999 300/" > wiki.200k.fr.vec 
head -n 200000 wiki.hr.vec | sed "1s/.*/199999 300/" > wiki.200k.hr.vec 
head -n 200000 wiki.it.vec | sed "1s/.*/199999 300/" > wiki.200k.it.vec 
head -n 200000 wiki.ru.vec | sed "1s/.*/199999 300/" > wiki.200k.ru.vec 
head -n 200000 wiki.tr.vec | sed "1s/.*/199999 300/" > wiki.200k.tr.vec 
