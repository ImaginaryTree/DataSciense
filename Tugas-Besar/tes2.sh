#!/bin/sh


while read line
do
	echo $line| tr '[M-ZA-Lm-za-l]' '[A-Za-z]' >> decrypted.txt
done < lyrics.txt
