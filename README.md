# StyleGAN
This is an implementation of a Photo2Style Generator Architecture for GAN on tensorflow. The model generates new image with refernce style or vice versa.

# Getting Started
* ([P2SNet.py](/libs/network/P2SNet.py), [config.py](/libs/configs/config.py)): These files are the main Photo to Style Generator network.

* ([datapipe.py](/datasets/datapipi.py)): This file's role is loading and changing to tensor your dataset taht are in photo2rtyle folder. you must put your training dataset in ([photo2style](/datasets/photo2style))

* ([pretrained_models](/pretrained_models)): you put the pretrained ([VGG 19](http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz)) model in this folder.

* ([train.py](/train.py)): this file is for training.
	To train this network
''' python train.py '''

* ([inference.py](/inference.py)): this file is for inference. we have 2 inference mode that are reference style and random style and you set True or False on --rand_style value.
	To run this file
''' python infernece.py --rand_style=True'''

# Result
<table >
    <tr >
    	<td><center>Photo</center></td>
        <td><center>Style</center></td>
        <td><center>Photo with style</center></td>
        <td><center>Style with Photo</center></td>
        <td><center>Photo with rand style</center></td>
        <td><center>Style with rand Photo</center></td>
    </tr>
    <tr>
    	<td>
    		<center><img src="/output/0.0009-1/oriA.jpg" height="280"></center>
    	</td>
    	<td>
    		<center><img src="/output/0.0009-1/oriB.jpg" height="280"></center>
    	</td>
        <td>
        	<center><img src="/output/0.0009-1/fake_AB.jpg" height="280"></center>
        </td>
        <td>
        	<center><img src="/output/0.0009-1/fake_BA.jpg" height="280"></center>
        </td>
	<td>
        	<center><img src="/output/0.0009-1/rand_AB.jpg" height="280"></center>
        </td>
        <td>
        	<center><img src="/output/0.0009-1/rand_BA.jpg" height="280"></center>
        </td>
    </tr>
</table>

