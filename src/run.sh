export PYTHONHASHSEED=0
find ../data/train_simple_part/* | xargs -L 1 -P 36 python etl_train.py 
find ../data/test_simple_part/* | xargs -L 1 -P 36 python etl_test.py 
