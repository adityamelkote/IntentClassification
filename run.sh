mkdir assets
#python main.py --n-epochs 10 --do-train
python main.py --n-epochs 10 --do-train --task custom --warmup-ratio .1
# python main.py --n-epochs 10 --do-train --task custom --reinit_n_layers 3
# python main.py --n-epochs 10 --do-train --task supcon --batch-size 64
