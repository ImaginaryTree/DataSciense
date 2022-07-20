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

echo -n "Masukan String : "
read sentence
echo "Unreverse : $sentence"
read -a split <<< "$sentence"
for word in ${split[@]}
do
	palindrome $word
	if [[ $? -eq 1 ]]
	then
		echo "$word = palindrome"
	else
		echo "$word = Not Palindrome"
	fi
done
