for lamb1 in $(seq 3.0 0.3 4.2)
do
    python -u run_itr.py --data pbmc --selector STG --lamb1 $lamb1 --permuted permuted --lr 1e-5 --lr2 1e-3 --w_dec1 0.01 --w_dec2 0.001 \

done

for lamb2 in $(seq 0.2 0.2 2.0)
do
    python -u run.py --data pbmc --selector HC --lamb1 0.0 --lamb2 $lamb2 --permuted permuted --lr 1e-5 --lr2 1e-3 --w_dec1 0.01 --w_dec2 0.001 \

done