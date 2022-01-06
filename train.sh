
set -xe

rm -rf /tmp/speech_commands_train/ /tmp/retrain_logs/; python train.py --data_bad=schlecht --data_good=gut --model_size_info 198 8 2 3 2 2 91 30 --dct_coefficient_count 36 --learning_rate 0.00007 --how_many_training_steps 100000 --validation_percentage 10 --batch_size 20

out=$1
if test -z "$out"; then
	 out=current.pb
fi
best=$(ls -1tr /tmp/speech_commands_train/best/crnn_*.meta | tail -n 1 | sed -e 's,\.meta,,')
python freeze.py --dct_coefficient_count 36  --output_file $out $best

python label_wav.py --graph $out --wav 'gut/*wav' > labels.txt
python label_wav.py --graph $out --wav 'schlecht/*wav' >> labels.txt
