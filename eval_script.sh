
#!/bin/bash

WEIGHTS_PATH=/usr/scratch/badile43/amarchei/checkpoints/


MVSEC_EVAL_SCRIPT=evals/eval_evs/eval_mvsec_evs.py
HKU_EVAL_SCRIPT=evals/eval_evs/eval_hku_evs.py

VAL_SPLIT_MVSEC=splits/mvsec/mvsec_val.txt
VAL_SPLIT_HKU=splits/hku/hku_val.txt

CONFIG_MVSEC=config/eval_mvsec.yaml
CONFIG_HKU=config/eval_hku.yaml

EXPNAME=DEVO_noctx

MODEL=DEVO
TRIALS=5
PLOT=true
PATCHIFIER_MODEL=gradual
DIM_FNET=64
DIM_INET=192



#### EVALUATING MODEL ON MVSEC DATASET

python $MVSEC_EVAL_SCRIPT \
        --config=$CONFIG_MVSEC \
        --weights=$WEIGHTS_PATH/$MODEL/150000.pth \
        --trials=$TRIALS \
        --plot=$PLOT \
        --expname=$EXPNAME \
        --model=$PATCHIFIER_MODEL \
        --val_split=$VAL_SPLIT_MVSEC \
        --dim_fnet=$DIM_FNET \
        --dim_inet=$DIM_INET


echo "Evaluation completed for $MODEL with $EXPNAME"


##### EVALUATING MODEL ON HKU DATASET


python $HKU_EVAL_SCRIPT \
        --config=$CONFIG \
        --weights=$WEIGHTS_PATH/$MODEL/150000.pth \
        --trials=$TRIALS \
        --plot=$PLOT \
        --expname=$EXPNAME \
        --val_split=$VAL_SPLIT \
        --model=$PATCHIFIER_MODEL \
        --dim_fnet=$DIM_FNET \
        --dim_inet=$DIM_INET