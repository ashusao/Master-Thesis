# Master-Thesis (Detecting Object &amp; its State SImultaneously)

## Abstract
This thesis jointly addresses the task of detecting the object and its state. The state
of an object is the vital information with many potential applications in surveil-
lance, anomaly detection, etc. Given an image, the task is to localize the objects
and identify its category and state which they represent. Traditionally the task
is usually addressed using handcrafted features. With the success of ConvNets,
all computer vision tasks are usually addressed using them. In this work, we will
describe several approaches for achieving the job using ConvNets. We investigated
the effect of end-to-end training, multi-task learning, several network architectures
to incorporate object class and geometry for state prediction. Our methods are up-
grades on state of the art Faster R-CNN network to simultaneously
detect both object and its state. In our ablation experiments, we also show that
state information is implicitly learned even when trained for object classification
only. Additionally, we found that state predictions are capable of generalizing to
novel object classes.
