The folders in this folder describes the architectures and their experimental results. Networks are intialized with Faster R-CNN models trained on Pascal VOC. For first 75k iterations, the
models are trained with the learning rate of 0.003 and then for next 25k iterations with 0.0003 which is same as used in Faster R-CNN. Training is performed in end to end manner with momentum update.

**Note:** The state prediction in all our architectures predicts the 10 class-state combinations of our dataset instead of 2 open and closed state because the state of one category is considered
to be diffrent from that of others.

## Performances

| Architectures | Results (mAP) |
| ------------- | ------------- |
| Baseline	|	32.3	|
| Sate-Specific | 	34.9	|
| Sibling Branch |	34.0	|
| Concat (fc7,cls) |	32.9	|
| Concat (fc7,cls,bbox) | 34.1		|
| Concat + FC (fc7,cls) |	34.1	|
| Concat + FC (fc7,cls,bbox) |	34.5		|
| FC + Elementwise (fc7, cls) |		34.1	|
| FC + Elementwise (fc7, cls, bbox) | 	33.9	|
| faster_cls_state	|	33.5	|
| faster_cls_state (finetune) |	23.2	|
| faster_cls_bbox_state |	31.7	|
| Spatial1 	|	32.4	|
| Spatial2	|	33.5	| 

### State-Specific Architecture

The best performing architecture was state-specific architecture in all our approaches which is a faster R-CNN network that 
considers each class-state combinations (such as openUmbrella, closedUmbrella) as a seperate class and the network is trained
to learn the features to distinguish between them.


