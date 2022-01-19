export OMP_NUM_THREADS=1
source $HOME/prod/kws/env2/bin/activate

rm -f stop

function listen {
 (  while ! test -f stop; do 
     echo "Starting $1"
     STREAM=$1 python listen.py --detection_threshold 1
     sleep 10
  done ) &
}

listen http://httpmedia.radiobremen.de/bremenvier.m3u
listen http://mp3channels.webradio.antenne.de/antenne
listen http://mp3channels.webradio.antenne.de/classic-rock-live
listen http://stream.104.6rtl.com/rtl-live/mp3-128
listen http://stream.blackbeatslive.de/
listen http://stream.ffn.de/ffn/mp3-192
listen http://stream.laut.fm/total-instrumental.m3u
listen http://stream.multicult.fm:8000/hifi.m3u
listen http://streams.br.de/b5aktuell_2.m3u
listen http://streams.br.de/bayern3_2.m3u
listen http://streams.planetradio.de/planetradio/mp3/hqlivestream.m3u
listen http://webstream.hitradion1.de/hitradion1
listen http://rs5.stream24.net:80/stream
listen http://www.fritz.de/live.m3u
listen http://www.ndr.de/resources/metadaten/audio/m3u/ndr2.m3u
listen http://www.ndr.de/resources/metadaten/audio/m3u/ndr903.m3u
listen http://www.antennebrandenburg.de/livemp3_s
listen http://streams.antennemv.de/80er/mp3-128/streams.antennemv.de/play.m3u
listen http://streams.br.de/brheimat_1.m3u
listen https://stream.rcs.revma.com/ypqt40u0x1zuv
listen http://streaming.fueralle.org:8000/frs-hi.mp3.m3u
listen https://www.deutschlandradio.de/streaming/dlf.m3u
listen http://direct.franceculture.fr/live/franceculture-midfi.mp3
listen http://mfmwr-022.ice.infomaniak.ch/mfmwr-022.mp3
listen http://mfmwr-019.ice.infomaniak.ch/mfmwr-019.mp3
listen https://scdn.nrjaudio.fm/adwz1/fr/30617/mp3_128.mp3
listen http://rock2000.stream.ouifm.fr/ouifmrock2000.mp3
listen https://stream.rfm.fr/rfm-wr6.mp3
listen https://hendrikjansen.nl/radioluisterlijst/catalonieradiostations.m3u
listen https://hendrikjansen.nl/radioluisterlijst/frieslandstreekzenders.m3u
listen https://hendrikjansen.nl/radioluisterlijst/kerstmis.m3u

sleep 2

wait

