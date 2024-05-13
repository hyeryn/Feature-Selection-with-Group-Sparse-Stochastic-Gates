for lamb2 in 100 1000 1500 2000
do
    python -u run.py --id 1 --data syn --dtype C --selector HC --distribution norm --lamb1 0.0 --lamb2 $lamb2 --permuted permuted \

done
