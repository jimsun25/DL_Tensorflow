:: Where the checkpoint and logs will be saved to.
:: TRAIN_DIR=./model/lenet-model

:: Where the dataset is saved to.
:: DATASET_DIR=./data/mnist

:: Download the dataset
python download_and_convert_data.py ^
  --dataset_name=mnist ^
  --dataset_dir=./data/mnist

:: Run training.
python train_image_classifier.py ^
  --train_dir=./model/lenet-model ^
  --dataset_name=mnist ^
  --dataset_split_name=train ^
  --dataset_dir=./data/mnist ^
  --model_name=lenet ^
  --preprocessing_name=lenet ^
  --max_number_of_steps=20000 ^
  --batch_size=50 ^
  --learning_rate=0.01 ^
  --save_interval_secs=60 ^
  --save_summaries_secs=60 ^
  --log_every_n_steps=100 ^
  --optimizer=sgd ^
  --learning_rate_decay_type=fixed ^
  --weight_decay=0

:: Run evaluation.
python eval_image_classifier.py ^
  --checkpoint_path=./model/lenet-model ^
  --eval_dir=./model/lenet-model ^
  --dataset_name=mnist ^
  --dataset_split_name=test ^
  --dataset_dir=./data/mnist ^
  --model_name=lenet
  