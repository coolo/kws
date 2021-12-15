C=19
LL=0.0001
DIR=work/C=$C,L=$LL,M1=633,M2=20,M3=2,M4=2,M5=248,M6=140,WS=28,ST=12/
mkdir $DIR
python train.py --data_bad=/space/coolo/data/schlecht/ --data_good=/space/coolo/data/gut/ --model_architecture crnn --model_size_info 63 20 2 2 2 2 248 140 --dct_coefficient_count $C --window_size_ms 28 --window_stride_ms 12 --learning_rate $LL --how_many_training_steps 20000 --summaries_dir $DIR/retrain_logs --train_dir $DIR/training --testing_percentage 0 --validation_percentage 10 | tee $DIR/output
