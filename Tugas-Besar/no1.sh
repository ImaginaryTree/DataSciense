#!/bin/sh
while true
do
result=0
cmd=clear
echo -n "Masukan Angka pertama : "
read result
while true
	do
	echo "=================="
	echo "   result = $result"
	echo "=================="
	echo
	echo "--------------------------"
	echo "|Menu Operator Aritmatika|"
	echo "--------------------------"
	echo "|  1. Tambah             |"
	echo "|  2. Kurang             |"
	echo "|  3. Kali               |"
	echo "|  4. Bagi               |"
	echo "|  5. Reset              |"
	echo "--------------------------"
	echo -n "Masukan Angka yag dipilih : "
	read choose

	case $choose in
	1)
		echo -n "Masukan Angka : "
		read angka
		result=$(expr $result + $angka)
	;;
	2)
		echo -n "Masukan Angka : "
		read angka
		result=$(expr $result - $angka)
	;;
	3)
		echo -n "Masukan Angka : "
		read angka
		let temp=$result*$angka
		result=$temp
	;;
	4)
		echo -n "Masukan Angka : "
		read angka
		result=$(expr $result / $angka)
	;;
	*)
		break
	;;
	esac
	$cmd
	done
$cmd
done
