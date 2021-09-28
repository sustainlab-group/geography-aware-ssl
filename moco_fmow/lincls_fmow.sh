#!/bin/bash
DATASET=fmow
PORT=10007

source /atlas/u/kayush/winter2020/SegFuture/segfutenv/bin/activate
cd /atlas/u/kayush/winter2020/jigsaw/moco_sat/moco_code

for LR in 0.3; do
for T in 0.07; do
BS=256
ARCH=resnet50
GPUS=2
# LOSS_TYPE=mc_cpc
LOSS_TYPE=cpc
MOCO_DIM=2048

for LIN_LR in 10; do

let PORT++
JOBNAME=${DATASET}_${ARCH}_${LOSS_TYPE}_${LR}_${BS}_${T}/${LIN_LR}

PT_DIR=./ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-${LR}_bs-${BS}_t-${T}_baseline

# WRAP="python main_lincls_cifar.py -a ${ARCH} --lr ${LIN_LR} --dist-url 'tcp://localhost:${PORT}' --multiprocessing-distributed --world-size 1 --rank 0 -j 4 --pretrained=${PT_DIR}/checkpoint_0999.pth.tar --save-dir ${PT_DIR}/lincls_${LIN_LR} /atlas/u/tsong/data/${DATASET}"
WRAP="python main_lincls_fmow.py -a ${ARCH} --lr ${LIN_LR} --dist-url 'tcp://localhost:${PORT}' --multiprocessing-distributed --world-size 1 --rank 0 -j 4 --pretrained=${PT_DIR}/checkpoint_0100.pth.tar --save-dir ${PT_DIR}/lincls_${LIN_LR} --data ${DATASET}"

echo ${WRAP}

sbatch --output=linout/%j.out --error=linout/%j.err \
    --exclude=atlas1,atlas2,atlas3,atlas4,atlas5,atlas6,atlas8,atlas13,atlas17,atlas18 \
    --nodes=1 --ntasks-per-node=1 --time=7-00:00:00 --mem=32G \
    --partition=atlas --cpus-per-task=16 \
    --gres=gpu:${GPUS} --job-name=${JOBNAME} --wrap="${WRAP}"

done
done
done