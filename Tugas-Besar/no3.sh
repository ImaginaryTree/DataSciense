#!/bin/sh

read -p "Masukan Nama File yang ingin di decrypt : " file
read -p "Masukan Nama File Hasil Decrypt : " file2
while read line
do
	echo $line| tr '[M-ZA-Lm-za-l]' '[A-Za-z]' >> $file2
done < $file
