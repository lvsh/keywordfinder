#!/bin/bash
baseDir="train"

for doc in `ls $baseDir/*.txt`
do
name=`basename $doc .txt`;
head -1 $doc > $baseDir"/"$name"-justTitle.txt";
done