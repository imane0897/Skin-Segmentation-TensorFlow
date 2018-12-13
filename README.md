# Skin-Segmentation-TensorFlow
This is a modified [SegNet](https://arxiv.org/abs/1511.00561) convolutional neural net for segmenting human skin from images.

## Main Idea:
The code emphasizes readability, simplicity and ease of understanding. It is meant to be looked at if you are starting out with TensorFlow and looking into building your own model. There are only two files, one for data loading, and one for the model definition, training and testing.

## Examples:
![alt text](https://ws3.sinaimg.cn/large/006tNbRwly1fy4uix95lvj30bv0hs3yt.jpg "Input image") ![alt text](https://ws4.sinaimg.cn/large/006tNbRwly1fy4ujdsm54j30bv0hs0st.jpg "Predicted segmentation")

Free stock image taken from [Pixabay](https://pixabay.com/)

## Explanation of Code Snippets:
Convolution is done with the tf.layers.conv2d layer, like so:
```python
def conv_with_bn(x, no_of_filters, kernel_size, training, strides=[1, 1], activation=tf.nn.relu, use_bias=True, name=None):
    conv = tf.layers.conv2d(x, no_of_filters, kernel_size, strides, padding='SAME', activation=activation,
                            use_bias=use_bias, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=name)
    conv = tf.layers.batch_normalization(conv, training=training)

    return conv
```
Downsampling is done again with the same convolutional layer, only with the strides changed to 2.

Upsampling is done via transpose convolutions:
```python
def trans_conv_with_bn(x, no_of_filters, kernel_size, training, strides=[2, 2], activation=tf.nn.relu, use_bias=True, name=None):
    conv = tf.layers.conv2d_transpose(x, no_of_filters, kernel_size, strides, padding='SAME', activation=activation,
                                      use_bias=use_bias, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=name)
    conv = tf.layers.batch_normalization(conv, training=training)
    return conv
```

The model itself is defined in the inference function:
```python
def inference(image_tensor, is_training):
    """Runs image through the network and returns predicted mask."""

    print('Building Network for Inference...')

    conv0 = conv_with_bn(image_tensor, 64, [3, 3], is_training, name='conv0')
    down0 = conv_with_bn(conv0, 64, [3, 3], is_training, [2, 2], name='down0')

    conv1 = conv_with_bn(down0, 128, [3, 3], is_training, name='conv1')
    down1 = conv_with_bn(conv1, 128, [3, 3], is_training, [2, 2], name='down1')

    conv2 = conv_with_bn(down1, 256, [3, 3], is_training, name='conv2')
    down2 = conv_with_bn(conv2, 256, [3, 3], is_training, [2, 2], name='down2')

    conv3 = conv_with_bn(down2, 512, [3, 3], is_training, name='conv3')
    down3 = conv_with_bn(conv3, 512, [3, 3], is_training, [2, 2], name='down3')

    up3 = trans_conv_with_bn(down3, 512, [3, 3], is_training, name='up3')
    unconv3 = conv_with_bn(up3, 512, [3, 3], is_training, name='unconv3')

    up2 = trans_conv_with_bn(unconv3, 256, [3, 3], is_training, name='up2')
    unconv2 = conv_with_bn(up2, 256, [3, 3], is_training, name='unconv2')

    up1 = trans_conv_with_bn(unconv2, 128, [3, 3], is_training, name='up1')
    unconv1 = conv_with_bn(up1, 128, [3, 3], is_training, name='unconv1')

    up0 = trans_conv_with_bn(unconv1, 64, [3, 3], is_training, name='up0')
    unconv0 = conv_with_bn(up0, 64, [3, 3], is_training, name='unconv0')

    pred = conv_with_bn(unconv0, NUM_CLASSES, [
                        3, 3], is_training, activation=None, use_bias=False, name='pred')

    print('Done, network built.')
    return pred
```
As can be seen, the model has 9 convolutional layers and calculates upto 512 feature maps. The architecture is simple to understand, the focus here is on readability.

## Dataset

[Pratheepan Dataset + Ground Truth](http://cs-chan.com/downloads_skin_dataset.html) by Chee Seng Chan is used. This dataset and groundtruth contains individual face photos and multiple face photos. 

## Result

```
Train loss: 0.16735801590494812
Train iou: 0.9361128197431564
Val. Loss: 0.38619453
Val. iou: 0.5982049
```

Training result model is saved in `model.zip` . Available in [Google Drive](https://drive.google.com/open?id=1XiyjcniyzKWp2j9FJKX2ofTDpW9SnOIp) cause it's over 100MB and git push is limited. 

Test result is given as follow.

```
amida-belly-dancer.png
0.7807401512136888
124511719065943_2.png
0.6977685433353673
2007_family.png
0.8747758540437838
Afishingfamilygettogether.png
0.7791820926126332
buck_family.png
0.9078863826232247
CDVP-group.png
0.8029181817884808
920480_f520.png
0.7522068378366268
07-c140-12family-red-rr-398h.png
0.7371939060381272
beck-family.png
0.8109091772227014
Aishwarya-Rai_20091229_aatheory.png
0.8798992545122576
06Apr03Face.png
0.8314519836655219
0520962400.png
0.8595130044405407
3115267-My-very-large-Indian-family-2.png
0.7720811597651617
chenhao0017me9.png
0.8188524671329502
abbasprize.png
0.7976665011702958

average
0.8068696998267575
```

