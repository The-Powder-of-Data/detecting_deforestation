# Notebooks
These Jupyter Notebooks are the approaches to the models made so far. <br>
Most of the work was done on Google Colab as a GPU makes a large difference for modle training
<br>
In the name of learning a new platform I wanted to have all my data on Google Cloud Platform. For these colab notebooks I have authenticated through my email with the project Google Cloud Buckets. If you would like to replicate this approach please setup a GCP account and you will get $300 for free for 80 days. ALternativly you could host your files locally. 


## 1. Multi Label Classification Notebook
The approach to this notebook was to create a model that can tag/label image tiles with observations. These results can then be used as layer 1 in our overall understanding and automation of processing satellite imagery of our earth.

### Data & Processing
As a start point we aquired a dataset ***insert link***. The origional 22gb torrent is no longer being seeded, however there is an alternative link to a balanced samples that you can download ***here***. <br>

For this example dataset you could choose not to do any pre processing as the satellie images come already broken into small enough tiles for preprocessing. If ingesting images from a [satellite API](https://www.programmableweb.com/news/top-10-satellites-apis/brief/2020/06/14) you would have to preprocess images into small enough patches/tiles in order for them to be ingested by the model. To see an example of this please look at the Semantic Segmentation Model approach. 

### Model, Results & Application
FastAI was a great libary once learning the basics. I was able to get great results using a few techiques that Fastai makes easy. 
- Disriminate Learning Rates
- Transfer learning
- Image sizing trick to increase the size of our dataset

## 2. Semantic Segmentation Notebook
Tensorflow was used as the main modeling libary for this notebook. Using transfer learning using a UNet architecutre with additional pooling yeilded decent results on the training dataset. I did also try a Resnet architecture with imagenet weights and found it yielded similar results. A large amount of learning around image data processing and how to evaluate a segmentation results went into this notebook. 
<br>

### Data Processing 
One of the biggest learning lessons of this project was image processing to ensure they are the right shape to be ingested by our designed model. As satellite images are high resolution objects we do not have powerful enough resources to run models over all pixels as our input. To accomidate for this, a large image has to be broken down into smaller patches (sometimes refered to as image tiles). Once both images and training masks are divided into patches the data can then be inputted into the modle for training. 
<br>
Once the model has been trained the smaller image patches need to be stitched back together to make a whole. This was done origionally manually and then using the libray of ***Patchify*** to assist in the process. A large shoutout to ***insert patchify video*** for updating the code to a libary that has not had updates in the last few years. 

- ***Insert image of patchify structure***

### Model, Results & Application 
Origionally this model was trained on a dataset of Dubai with 6 labels. After tuning the model to satisfactory levels, I experimented with predicting segmentation masks on BC satellite images. Even though this model was trained on a different context (Dubai) than what its intended application was to be (logging in BC) it is a starting step to this continued experiment. To my suprise it was able to pick out features such as clear cut forest quite well. The next steps would be to aquire training mask data for the intended region of use and compare results. 

