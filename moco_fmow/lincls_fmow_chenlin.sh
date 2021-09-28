#!/bin/bash
DATASET=fmow
PORT=10008

source /atlas/u/kayush/winter2020/SegFuture/segfutenv/bin/activate
cd /atlas/u/kayush/winter2020/jigsaw/moco_sat/moco_code

for LR in 0.3; do
for T in 0.07; do
BS=16
ARCH=resnet50
GPUS=4
# LOSS_TYPE=mc_cpc
LOSS_TYPE=cpc
MOCO_DIM=128

for LIN_LR in 0.001; do

let PORT++
JOBNAME=${DATASET}_${ARCH}_${LOSS_TYPE}_${LR}_${BS}_${T}/${LIN_LR}

#PT_DIR=/atlas/u/kayush/winter2020/jigsaw/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-${LR}_bs-${BS}_t-${T}_mocodim-128_baseline_224_scale_0.7
#DIR=/atlas/u/chenlin/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-${LR}_bs-${BS}_t-${T}_mocodim-128_baseline_224_scale_0.7

#PT_DIR=/atlas/u/kayush/winter2020/jigsaw/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-${LR}_bs-${BS}_t-${T}_mocodim-128_baseline_224_scale_0.2
#DIR=/atlas/u/chenlin/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-${LR}_bs-${BS}_t-${T}_mocodim-128_baseline_224_scale_0.2

#PT_DIR=/atlas/u/kayush/winter2020/jigsaw/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-${LR}_bs-${BS}_t-${T}_mocodim-128_temporal_224_with_add_transform
#DIR=/atlas/u/chenlin/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-${LR}_bs-${BS}_t-${T}_mocodim-128_temporal_224_with_add_transform

#PT_DIR=/atlas/u/kayush/winter2020/jigsaw/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-${LR}_bs-${BS}_t-${T}_mocodim-128_temporal_224_no_add_transform
#DIR=/atlas/u/chenlin/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-${LR}_bs-${BS}_t-${T}_mocodim-128_temporal_224_no_add_transform

#PT_DIR=/atlas/u/kayush/winter2020/jigsaw/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-${LR}_bs-${BS}_t-${T}_temporal_224
#DIR=/atlas/u/chenlin/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-${LR}_bs-${BS}_t-${T}_temporal_224

# NEW baseline
#PT_DIR=//atlas/u/kayush/winter2020/jigsaw/moco_sat/moco_code/ckpt/fmow/resnet50/cpc_500/lr-0.03_bs-256_t-0.02_mocodim-128_baseline_224_scale_0.2_comparable_448
#DIR=/atlas/u/chenlin/moco_sat/moco_code/ckpt/fmow/resnet50/cpc_500/lr-0.03_bs-256_t-0.02_mocodim-128_baseline_224_scale_0.2_comparable_448

# NEW temporal *********
# PT_DIR=/atlas/u/kayush/winter2020/jigsaw/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_temporal_224_exactly_same_as_32x32_with_add_transform
# DIR=/atlas/u/kayush/winter2020/jigsaw/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_temporal_224_exactly_same_as_32x32_with_add_transform_200

#PT_DIR=/atlas/u/kayush/winter2020/jigsaw/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_temporal_224_exactly_same_as_baseline
#DIR=/atlas/u/chenlin/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_temporal_224_exactly_same_as_baseline

#PT_DIR=/atlas/u/kayush/winter2020/jigsaw/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_temporal_224_exactly_same_as_32x32
#DIR=/atlas/u/chenlin/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_temporal_224_exactly_same_as_32x32

#PT_DIR=/atlas/u/kayush/winter2020/jigsaw/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_baseline_224_scale_0.2
#DIR=/atlas/u/chenlin/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_baseline_224_scale_0.2



#PT_DIR=/atlas/u/kayush/winter2020/jigsaw/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_baseline_224_scale_0.2_geohead_corrected
#DIR=/atlas/u/chenlin/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_baseline_224_scale_0.2_geohead_corrected

#PT_DIR=/atlas/u/kayush/winter2020/jigsaw/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_baseline_224_scale_0.2_comparable_448_geohead_corrected
#DIR=/atlas/u/chenlin/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_baseline_224_scale_0.2_comparable_448_geohead_corrected

#PT_DIR=/atlas/u/kayush/winter2020/jigsaw/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_temporal_224_exactly_same_as_32x32_with_add_transform_geohead_corrected
#DIR=/atlas/u/chenlin/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_temporal_224_exactly_same_as_32x32_with_add_transform_geohead_corrected


# epoch 200, original baseline **********
# PT_DIR=/atlas/u/kayush/winter2020/jigsaw/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_baseline_original_without_initial_resize
# DIR=/atlas/u/kayush/winter2020/jigsaw/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_baseline_original_without_initial_resize_200

## subset 224 448 resolution
#PT_DIR=/atlas/u/chenlin/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_image_224_448_original_with_initial_resize_to_224
#DIR=/atlas/u/chenlin/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_image_224_448_original_with_initial_resize_to_224

#PT_DIR=/atlas/u/chenlin/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_image_224_448_comparable_448
#DIR=/atlas/u/chenlin/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_image_224_448_comparable_448

#PT_DIR=/atlas/u/chenlin/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_image_224_448_original_without_initial_resize_to_224_448
#DIR=/atlas/u/chenlin/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_image_224_448_original_without_initial_resize_to_224_448

# PT_DIR=/atlas/u/chenlin/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_image_224_448_comparable_672
# DIR=/atlas/u/chenlin/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_image_224_448_comparable_672_100


# epoch 200 ************* geo+temp
# PT_DIR=/atlas/u/kayush/winter2020/jigsaw/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_temporal_224_exactly_same_as_32x32_with_add_transform_geohead_corrected
# DIR=/atlas/u/kayush/winter2020/jigsaw/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_temporal_224_exactly_same_as_32x32_with_add_transform_geohead_corrected_200

# PT_DIR=/atlas/u/kayush/winter2020/jigsaw/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_baseline_224_scale_0.2_comparable_448_geohead_corrected
# DIR=/atlas/u/kayush/winter2020/jigsaw/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_baseline_224_scale_0.2_comparable_448_geohead_corrected_200

#PT_DIR=/atlas/u/kayush/winter2020/jigsaw/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_baseline_224_scale_0.2_geohead_corrected
#DIR=/atlas/u/chenlin/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_baseline_224_scale_0.2_geohead_corrected_200

#PT_DIR=/atlas/u/kayush/winter2020/jigsaw/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_geo_baseline_without_initial_resize
#DIR=/atlas/u/chenlin/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_geo_baseline_without_initial_resize_200

#PT_DIR=/atlas/u/kayush/winter2020/jigsaw/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_baseline_224_scale_0.2
#DIR=/atlas/u/chenlin/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_baseline_224_scale_0.2_200

#PT_DIR=/atlas/u/kayush/winter2020/jigsaw/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_temporal_baseline_without_intial_resize
#DIR=/atlas/u/chenlin/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_temporal_baseline_without_intial_resize_100

####### imagenet moco start weighths for finetuning
# PT_DIR=/atlas/u/buzkent/ImageNet/unsupervised_checkpoints/224x224_ckpt_0199.pth.tar
# DIR=/atlas/u/kayush/winter2020/jigsaw/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/224x224_ckpt_0199

PT_DIR=/atlas/u/buzkent/ImageNet/unsupervised_checkpoints/224x224_ckpt_0199.pth.tar
DIR=/atlas/u/kayush/winter2020/jigsaw/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/224x224_ckpt_0199_whole_finetune


# PT_DIR=/atlas/u/kayush/winter2020/jigsaw/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_imagenet_plus_temp+geo/checkpoint_0212.pth.tar
# DIR=/atlas/u/kayush/winter2020/jigsaw/moco_sat/moco_code/ckpt/${DATASET}/${ARCH}/${LOSS_TYPE}_500/lr-0.03_bs-256_t-0.02_mocodim-128_imagenet_plus_temp+geo


# WRAP="python main_lincls_cifar.py -a ${ARCH} --lr ${LIN_LR} --dist-url 'tcp://localhost:${PORT}' --multiprocessing-distributed --world-size 1 --rank 0 -j 4 --pretrained=${PT_DIR}/checkpoint_0999.pth.tar --save-dir ${PT_DIR}/lincls_${LIN_LR} /atlas/u/tsong/data/${DATASET}"
# WRAP="python main_lincls_fmow_chenlin.py -a ${ARCH} --lr ${LIN_LR} --dist-url 'tcp://localhost:${PORT}' --multiprocessing-distributed --world-size 1 --rank 0 -j 4 --pretrained=${PT_DIR}/checkpoint_0200.pth.tar --save-dir ${DIR}/lincls_${LIN_LR} --data ${DATASET} --batch-size 256"
WRAP="python main_lincls_fmow_chenlin.py -a ${ARCH} --lr ${LIN_LR} --dist-url 'tcp://localhost:${PORT}' --multiprocessing-distributed --world-size 1 --rank 0 -j 4 --pretrained=${PT_DIR} --save-dir ${DIR}/lincls_${LIN_LR} --data ${DATASET} --batch-size 256"


echo ${WRAP}

sbatch --output=linout_chenlin/%j.out --error=linout_chenlin/%j.err \
    --exclude=atlas1,atlas2,atlas3,atlas4,atlas5,atlas6,atlas18,atlas13,atlas16,atlas17 \
    --nodes=1 --ntasks-per-node=1 --time=7-00:00:00 --mem=32G \
    --partition=atlas --cpus-per-task=16 \
    --gres=gpu:${GPUS} --job-name=${JOBNAME} --wrap="${WRAP}"

done
done
done