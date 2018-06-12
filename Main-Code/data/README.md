This directory holds (*after you download them*):
- Caffe models pre-trained on ImageNet
- Faster R-CNN models
- sample of Detection Dataset with annotated state information

To download Caffe models (ZF, VGG16) pre-trained on ImageNet, run:

```
./data/scripts/fetch_imagenet_models.sh
```

This script will populate `data/imagenet_models`.

To download Faster R-CNN models trained on VOC 2007, run:

```
./data/scripts/fetch_faster_rcnn_models.sh
```

This script will populate `data/faster_rcnn_models`.

The sample Dataset is described in `data/stateDetection/` folder which consists of:
- `./data/stateDetection/JPEGImages/`: contains the sample images of dataset.
- `./data/stateDetection/Annotations/`: contains the annotations of images in VOC format where state information
	is captured in objects name tag. for e.g. <name>umbrella_openUmbrella</name> where 'umbrella' represents the object category
	and 'openUmbrella' describes its state.
- `./data/stateDetection/ImageSets/`: contains train/val .txt which describes the image id in training and validation set.
