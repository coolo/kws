python train.py --data_bad=/space/coolo/data/schlecht/ --data_good=/space/coolo/data/gut/ --model_architecture crnn --model_size_info 187 13 4 2 2 2 300 300 --dct_coefficient_count 47 --window_size_ms 20 --window_stride_ms 10 --learning_rate 0.0002 --how_many_training_steps 15000 --summaries_dir work/C=47,L=0.0002,M1=187,M2=13,M3=4,M4=2,M5=300,M6=300,WS=20,ST=10,BS=36/retrain_logs --train_dir work/C=47,L=0.0002,M1=187,M2=13,M3=4,M4=2,M5=300,M6=300,WS=20,ST=10,BS=36/training --testing_percentage 0 --validation_percentage 10 --batch_size 36
