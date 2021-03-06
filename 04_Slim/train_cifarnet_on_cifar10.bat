:: Where the checkpoint and logs will be saved to.
:: TRAIN_DIR=./model/cifarnet-model

:: Where the dataset is saved to.
:: DATASET_DIR=./data/cifar10

:: Download the dataset
python download_and_convert_data.py ^
  --dataset_name=cifar10 ^
  --dataset_dir=./data/cifar10

:: Run training.
python train_image_classifier.py ^
  --train_dir=./model/cifarnet-model ^
  --dataset_name=cifar10 ^
  --dataset_split_name=train ^
  --dataset_dir=./data/cifar10 ^
  --model_name=cifarnet ^
  --preprocessing_name=cifarnet ^
  --max_number_of_steps=100000 ^
  --batch_size=128 ^
  --save_interval_secs=120 ^
  --save_summaries_secs=120 ^
  --log_every_n_steps=100 ^
  --optimizer=sgd ^
  --learning_rate=0.1 ^
  --learning_rate_decay_factor=0.1 ^
  --num_epochs_per_decay=200 ^
  --weight_decay=0.004

:: Run evaluation.
python eval_image_classifier.py ^
  --checkpoint_path=./model/cifarnet-model ^
  --eval_dir=./model/cifarnet-model ^
  --dataset_name=cifar10 ^
  --dataset_split_name=test ^
  --dataset_dir=./data/cifar10 ^
  --model_name=cifarnet
