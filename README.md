## Geography-Aware Self-Supervised Learning (ICCV 2021)
[**Project**](https://geography-aware-ssl.github.io/) | [**Paper**](https://arxiv.org/abs/2011.09980) | [**Poster**](https://geography-aware-ssl.github.io/static/images/ICCV%202021%20Poster.png)



[Kumar Ayush](https://kayush95.github.io)<sup>\*</sup>, [Burak Uzkent](https://uzkent.github.io/)<sup>\*</sup>, [Chenlin Meng](https://cs.stanford.edu/~chenlin/)<sup>\*</sup>, [Kumar Tanmay](), [Marshall Burke](https://web.stanford.edu/~mburke/), [David Lobell](https://earth.stanford.edu/people/david-lobell), [Stefano Ermon](https://cs.stanford.edu/~ermon/).
<br> Stanford University
<br>In [ICCV](https://arxiv.org/abs/2011.09980), 2021.

<p align="center">
  <img src="https://raw.githubusercontent.com/sustainlab-group/geography-aware-ssl/main/.github/images/ap2.png" width="500">
</p>


This is a PyTorch implementation of [Geography-Aware Self-Supervised Learning](https://arxiv.org/abs/2011.09980). We use the the official implementation of <a href="https://github.com/facebookresearch/moco">MoCo-v2</a> for developing our methods.

### fMoW Dataset

<a href="https://arxiv.org/abs/1711.07846">Functional Map of the Dataset</a> can be downloaded from their website/repo.
You can create csvs similar to the ones in the `csvs/` folder.

<p align="center">
  <img src="https://raw.githubusercontent.com/sustainlab-group/geography-aware-ssl/main/.github/images/fmow_coords.png" width="500">
</p>

Map showing distribution of the fMoW dataset.

### Preparation

Install PyTorch and download the fMoW dataset.

### Self-Supervised Training

Similar to official implementation of MoCo-v2, this implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

To do self-supervised pre-training of a ResNet-50 model on fmow using our MoCo-v2+Geo+TP model in an 4-gpu machine, run:
```
python moco_fmow/main_moco_geo+tp.py \ 
    -a resnet50 \
    --lr 0.03 \
    --dist-url 'tcp://localhost:14653' --multiprocessing-distributed --moco-t 0.02 --world-size 1 --rank 0 --mlp -j 4 \
    --loss cpc --epochs 200 --batch-size 256 --moco-dim 128 --aug-plus --cos \
    --save-dir ${PT_DIR} \
    --data fmow
```

To do self-supervised pre-training of a ResNet-50 model on fmow using our MoCo-v2+TP model in an 4-gpu machine, run:
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
<td align="center">200</td>
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
**Download the GeoImageNet** - The instructions to download GeoImageNet set are given <a href="https://github.com/sustainlab-group/geography-aware-ssl/tree/main/geoimagenet_downloader">here</a>. Using this repository, we can download in the order of 2M images together with their coordinates. In the paper, we use 540k images for the GeoImageNet. The download process should download the images into their representative class folder.

**Clustering** - Once, we download the GeoImageNet dataset, we can use a clustering algorithm to cluster the images using their geo-coordinates. In the paper, we use K-means clustering to cluster 540k images into 100 clusters, however, any clustering algorithm can be used. After K-means clustering, we need to create a csv file similar to ones in the **./csvs/** folder.

**Perform Self-Supervised Learning** - After downloading the GeoImageNet and clustering the images, we can perform self-supervised learning. To do it, you can execute the following command :
```
python moco_fmow/main_moco_geo+tp.py \ 
    -a resnet50 \
    --lr 0.03 \
    --dist-url 'tcp://localhost:14653' --multiprocessing-distributed --moco-t 0.02 --world-size 1 --rank 0 --mlp -j 4 \
    --loss cpc --epochs 200 --batch-size 256 --moco-dim 128 --aug-plus --cos \
    --save-dir ${PT_DIR}
```

**Linear Classification** - After learning the representations with MoCo-v2-geo, we can train the linear layer to classify GeoImageNet images. With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 4-gpu machine, run:
```
python moco_fmow/main_lincls.py \
    -a resnet50 \
    --lr 1 \
    --dist-url 'tcp://localhost:14653' --multiprocessing-distributed --world-size 1 --rank 0 -j 4 \
    --pretrained=${PT_DIR} \
    --save-dir ${PTDIR}/lincls \
    --batch-size 256
```

### Transfer Learning Experiments
We use Retina-Net implementation from this <a href="https://github.com/yhenon/pytorch-retinanet">repository</a> for object detection experiments on xView. We use PSANet implementation from this <a href="https://github.com/hszhao/semseg">repository</a> for semantic segmentation experiments on SpaceNet.


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


