<br/>
<h1 align="center">Capstone : Panoptic Segmentation with DETR
<br/>
<!-- toc -->
    <br>
    
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/RajamannarAanjaram/badges/)
[![Awesome Badges](https://img.shields.io/badge/badges-awesome-green.svg)](https://github.com/RajamannarAanjaram/badges)
    <br>
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://GitHub.com/RajamannarAanjaram/)


<!-- toc -->

## Questions

As a part of the CAPSTONE project you need to explain missing:
 - We take the encoded image (dxH/32xW/32) and send it to Multi-Head Attention **(FROM WHERE DO WE TAKE THIS ENCODED IMAGE?)**
 - We also send dxN Box embeddings to the Multi-Head Attention
 - We do something here to generate NxMxH/32xW/32 maps. **(WHAT DO WE DO HERE?)**
 - Then we concatenate these maps with Res5 Block **(WHERE IS THIS COMING FROM?)**


#### FROM WHERE DO WE TAKE THIS ENCODED IMAGE?

We take this encoded image from a Convolution backbone network like ResNet-50 or ResNet-101.  The image is passed through the network which is used as a feature extractor. We remove the last few layers from RestNet along with the GAP & Classification layers before using it as a feature extractor.

The extracted feature-map is then passed through a transition layer to reduce the dimension(in the channel dimension) and finally used as  \the enoded image along with positonal encoding for the transformer.
    
It can also be seen that the encoder is encoding the input image into query-key-value pairs, and the key-value pairs are passed to the decoder for cross-attention.

#### WHAT DO WE DO HERE to generate NxMxH/32xW/32 maps ?

We use the Cross Attention scores from the last decoder layer for every Object Query (Object Embedding) and overlay with the encoded input from the encoder. The attention focuses on the corner of each to object to predict tight bounding box. The result is that each object query (N) produces Attention feature maps which we use further for panoptic segmentation.

#### Where is the Res5 Block coming from which we use to concatenate ?

The Res5/Res4/Res3/Res2 blocks are coming the backbone convolution networks we used initially to encode our input image. The concept is similar to the U-NET whereby we use it to upsample the encoded images to get mask over the whole image.


## Approach

To solve the problem of panoptic segmentation with DETR for a custom dataset, the following approach has been planned :

1. Since our custom dataset has only one label per image, we will use a pretrained DETR model on COCO to annotate the image.
2. Once the image is annotated with COCO lables, we will overlay it with our Custom Annotate for the required class.
3. Next, we wil create Bounding Boxes in the image using the segementation masks. This finally prepares our dataset. 
4. Finally split the data into train & test set in 80-20 split.
5. For Model training, I plan to use HuggingFace transformer library.
6. First the Object Detection is trained and we save the checkpoint , on the custom dataset.
7. Then we load this checkpoint for the segmentation training by freezing the backbone and detection head. This trains the segmentation head.
8. Once the segmentation head is trained, then we unfreeze the model, and train the whole thing with a lower learning rate for the model to stabilise and adjust.
9. Finally the new model is usable for inference in the new custom dataset.
