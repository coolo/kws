#DNN Models
python train.py --model_architecture dnn --model_size_info 436 436 436 --dct_coefficient_count 10 --window_size_ms 40 --window_stride_ms 20 --learning_rate 0.0005,0.0001,0.00002 --how_many_training_steps 10000,10000,10000 --summaries_dir work/DNN/DNN3/retrain_logs --train_dir work/DNN/DNN3/training  --data_bad=/space/coolo/data/schlecht/ --data_good=/space/coolo/data/gut/ --testing_percentage 5 --validation_percentage 5

#CNN Models
python train.py --model_architecture cnn --model_size_info 60 10 4 1 1 76 10 4 2 1 58 128 --dct_coefficient_count 10 --window_size_ms 40 --window_stride_ms 20 --learning_rate 0.0005,0.0001,0.00002 --how_many_training_steps 10000,10000,10000 --summaries_dir work/CNN/CNN3/retrain_logs --train_dir work/CNN/CNN3/training --data_bad=/space/coolo/data/schlecht/ --data_good=/space/coolo/data/gut/ --testing_percentage 5 --validation_percentage 5

#Basic LSTM Models
python train.py --model_architecture basic_lstm --model_size_info 344 --dct_coefficient_count 10 --window_size_ms 40 --window_stride_ms 20 --learning_rate 0.0005,0.0001,0.00002 --how_many_training_steps 10000,10000,10000 --summaries_dir work/Basic_LSTM/Basic_LSTM3/retrain_logs --train_dir work/Basic_LSTM/Basic_LSTM3/training --data_bad=/space/coolo/data/schlecht/ --data_good=/space/coolo/data/gut/ --testing_percentage 5 --validation_percentage 5

#LSTM Models
python train.py --model_architecture lstm --model_size_info 188 500 --dct_coefficient_count 10 --window_size_ms 40 --window_stride_ms 20 --learning_rate 0.0005,0.0001,0.00002 --how_many_training_steps 10000,10000,10000 --summaries_dir work/LSTM/LSTM3/retrain_logs --train_dir work/LSTM/LSTM3/training --data_bad=/space/coolo/data/schlecht/ --data_good=/space/coolo/data/gut/ --testing_percentage 5 --validation_percentage 5

#GRU Models
python train.py --model_architecture gru --model_size_info 1 400 --dct_coefficient_count 10 --window_size_ms 40 --window_stride_ms 20 --learning_rate 0.0005,0.0001,0.00002 --how_many_training_steps 10000,10000,10000 --summaries_dir work/GRU/GRU3/retrain_logs --train_dir work/GRU/GRU3/training --data_bad=/space/coolo/data/schlecht/ --data_good=/space/coolo/data/gut/ --testing_percentage 5 --validation_percentage 5

#CRNN Models
python train.py --model_architecture crnn --model_size_info 100 10 4 2 1 2 136 188 --dct_coefficient_count 10 --window_size_ms 40 --window_stride_ms 20 --learning_rate 0.0005,0.0001,0.00002 --how_many_training_steps 10000,10000,10000 --summaries_dir work/CRNN/CRNN3/retrain_logs --train_dir work/CRNN/CRNN3/training --data_bad=/space/coolo/data/schlecht/ --data_good=/space/coolo/data/gut/ --testing_percentage 5 --validation_percentage 5

#DS-CNN Models
python train.py --model_architecture ds_cnn --model_size_info 6 276 10 4 2 1 276 3 3 2 2 276 3 3 1 1 276 3 3 1 1 276 3 3 1 1 276 3 3 1 1 --dct_coefficient_count 10 --window_size_ms 40 --window_stride_ms 20 --learning_rate 0.0005,0.0001,0.00002 --how_many_training_steps 10000,10000,10000 --summaries_dir work/DS_CNN/DS_CNN3/retrain_logs --train_dir work/DS_CNN/DS_CNN3/training --data_bad=/space/coolo/data/schlecht/ --data_good=/space/coolo/data/gut/ --testing_percentage 5 --validation_percentage 5
