python train.py --data_bad=/space/coolo/data/schlecht/ --data_good=/space/coolo/data/gut/ --model_architecture crnn --model_size_info 40 16 6 2 2 2 300 300 --dct_coefficient_count 50 --window_size_ms 50 --window_stride_ms 10 --learning_rate 0.0001 --how_many_training_steps 15000 --summaries_dir work/C=50,L=0.0001,M1=40,M2=16,M3=6,M4=2,M5=300,M6=300,WS=50,ST=10,BS=17/retrain_logs --train_dir work/C=50,L=0.0001,M1=40,M2=16,M3=6,M4=2,M5=300,M6=300,WS=50,ST=10,BS=17/training --testing_percentage 0 --validation_percentage 10 --batch_size 17