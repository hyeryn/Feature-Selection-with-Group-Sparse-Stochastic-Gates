for lamb1 in $(seq 0.0 1000.0 7000.0)
do
    python -u run.py --data gas --selector HC --distribution norm --lamb1 $lamb1 --lamb2 13000 \

done

for lamb1 in $(seq 0.0 1000.0 7000.0)
do
    python -u run.py --data gas --selector HC --distribution origin --lamb1 $lamb1 --lamb2 13000 \

done