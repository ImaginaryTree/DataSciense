#!/bin/sh

palindrome(){
	get=$1
	len=${#get}
	for (( i=0;i<len/2;i+=1 ))
	do
		if [[ ${get:i:1} != ${get:len-i-1:1} ]]
		then
			return 0
		fi
	done
	return 1
}

Tpal="Palindrome "
Fpal="!Palindrome "
while read sentence
do
	check=""
	read -a split <<< "$sentence"
	for word in ${split[@]}
	do
		palindrome $word
		if [[ $? -eq 1 ]]
		then
			check+="${Tpal}"
		else
			check+="${Fpal}"
		fi
	done
	echo "Success"
	echo $check >> result.txt
done < palchecker.txt
