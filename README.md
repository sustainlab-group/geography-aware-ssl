## Geography-Aware Self-Supervised Learning (ICCV 2021)

<p align="center">
  <img src="https://raw.githubusercontent.com/sustainlab-group/geography-aware-ssl/main/.github/images/ap2.png" width="500">
</p>


This is a PyTorch implementation of **Geography-Aware Self-Supervised Learning** (https://arxiv.org/abs/2011.09980). We use the the official implementation of <a href="https://github.com/facebookresearch/moco">MoCo-v2</a> for developing our methods.

 * [Project Page](https://geography-aware-ssl.github.io/)
 * [Paper](https://arxiv.org/pdf/2011.09980.pdf)
 * [Poster](https://geography-aware-ssl.github.io/static/images/ICCV%202021%20Poster.png)

### fMoW Dataset

<a href="https://arxiv.org/abs/1711.07846">Functional Map of the Dataset</a> can be downloaded from their website/repo.
You can create csvs similar to the ones in the `csvs/` folder.

### Unsupervised Training

Similar to official implementation of MoCo-v2, this implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

To do unsupervised pre-training of a ResNet-50 model on fmow using our MoCo-v2+Geo+TP model in an 4-gpu machine, run:
```
python moco_fmow/main_moco_geo+tp.py \ 
    -a resnet50 \
    --lr 0.03 \
    --dist-url 'tcp://localhost:14653' --multiprocessing-distributed --moco-t 0.02 --world-size 1 --rank 0 --mlp -j 4 \
    --loss cpc --epochs 200 --batch-size 256 --moco-dim 128 --aug-plus --cos \
    --save-dir ${PT_DIR} \
    --data fmow
```

To do unsupervised pre-training of a ResNet-50 model on fmow using our MoCo-v2+TP model in an 4-gpu machine, run:
```
python moco_fmow/main_moco_tp.py \ 
    -a resnet50 \
    --lr 0.03 \
    --dist-url 'tcp://localhost:14653' --multiprocessing-distributed --moco-t 0.02 --world-size 1 --rank 0 --mlp -j 4 \
    --loss cpc --epochs 200 --batch-size 256 --moco-dim 128 --aug-plus --cos \
    --save-dir ${PT_DIR} \
    --data fmow
```

### Linear Classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 4-gpu machine, run:
```
python moco_fmow/main_lincls.py \
    -a resnet50 \
    --lr 1 \
    --dist-url 'tcp://localhost:14653' --multiprocessing-distributed --world-size 1 --rank 0 -j 4 \
    --pretrained=${PT_DIR} \
    --save-dir ${PTDIR}/lincls \
    --data fmow --batch-size 256
```
### Models

Our pre-trained ResNet-50 models can be downloaded as following:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">epochs</th>
<th valign="bottom">model</th>
<!-- TABLE BODY -->
<tr><td align="left">MoCo-v2</td>
<td align="center"><a href="">download</a></td>
</tr>
<tr><td align="left">MoCo-v2-Geo</td>
<td align="center">200</td>
<td align="center"><a href="">download</a></td>
</tr>
</tr>
<tr><td align="left">MoCo-v2-TP</td>
<td align="center">200</td>
<td align="center"><a href="">download</a></td>
</tr>
<tr><td align="left">MoCo-v2+Geo+TP</td>
<td align="center">200</td>
<td align="center"><a href="">download</a></td>
</tr>
</tbody></table>

### GeoImageNet
The instructions to download GeoImageNet dataset are given <a href="https://github.com/sustainlab-group/geography-aware-ssl/tree/main/geoimagenet_downloader">here</a>.

### Transfer Learning Experiments
We use Retina-Net implementation from this <a href="https://github.com/yhenon/pytorch-retinanet">repository for object detection experiments on xView. We use PSANet implementation from this <a href="https://github.com/hszhao/semseg">repository for semantic segmentation experiments on SpaceNet.


### Citing
If you find our work useful, please consider citing:
```
@article{ayush2021geography,
      title={Geography-Aware Self-Supervised Learning},
      author={Ayush, Kumar and Uzkent, Burak and Meng, Chenlin and Tanmay, Kumar and Burke, Marshall and Lobell, David and Ermon, Stefano},
      journal={ICCV},
      year={2021}
    }
```


