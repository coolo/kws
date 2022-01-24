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
			if test -f schlecht/$file; then
			    mv $file schlecht/$RANDOM.wav
			else
			    mv -n $file schlecht
			fi
			break
			;;
		g)
			mv -n $file gut || mv -n file gut/$RANDOM.wav
			break
			;;
		n)	
			continue
			;;
        esac
    done
done
