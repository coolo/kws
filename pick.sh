for file in $(ls -1 *.wav); do
    while true; do 
	aplay $file
	echo "(g)ut (s)chlecht (n)ochmal (i)gnore"
	input=
	read -rsn1 input
	case $input in
		i)
			mv $file ignored
			break
			;;
		s)
			mv -n $file schlecht
			break
			;;
		g)
			mv -n $file gut
			break
			;;
		n)	
			continue
			;;
        esac
    done
done
