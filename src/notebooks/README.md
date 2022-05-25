# Notebooks Intro
These Jupyter Notebooks are the approaches to the models made so far. <br>
Most of the work was done on Google Colab as a GPU makes a large difference for model training
<br>
In the name of learning a new platform I wanted to have all my data on Google Cloud Platform. For these colab notebooks I have authenticated through my email with the project Google Cloud Buckets. If you would like to replicate this approach please setup a GCP account and you will get $300 for free for 80 days. Alternatively you could host your files locally.
 
My vision for this project was to use combine my Data Science learning and Satellite Imagery to build an ***Analytics Time Machine***. For the short duration of this project (2 weeks) I focused on the first 2 components.
 
![Project Vision](../../output/content/vision_1.PNG)
 
<br>
 
## 1. **Multi Label Classification**
The approach to this notebook was to create a model that can tag/label image tiles with observations. These results can then be used as layer 1 in our overall understanding and automation of processing satellite imagery of our earth.
 
![MLC Example](../../output/content/MLC_example_1.png)
 
 
### **Data & Processing**
As a start point I used the [Planet dataset](https://www.kaggle.com/competitions/planet-understanding-the-amazon-from-space/data). The original 22gb torrent is no longer being seeded, however there is an [alternative link](https://www.kaggle.com/datasets/nikitarom/planets-dataset) to a balanced samples that you can download. <br>
 
For this example dataset you could choose not to do any pre processing as the satellite images come already broken into small enough tiles for preprocessing. If ingesting images from a [satellite API](https://www.programmableweb.com/news/top-10-satellites-apis/brief/2020/06/14) you would have to preprocess images into small enough patches/tiles in order for them to be ingested by the model. To see an example of this please look at the Semantic Segmentation Model approach.
<br> <br>
 
### **Model, Results & Application**
FastAI was a great library once learning the basics. I was able to get strong results using a few techniques.
- Discriminate Learning Rates
- Transfer learning
- Image sizing trick to increase the size of our dataset
<br>
 
After a variety of iterations the max result obtained was an f2 of 0.93 which puts this model in the top 60 results of 888 entries in the original competition. 
 
![MLC Results](../../output/content/MLC_result.jpg)
 
<br>
 
## 2. **Semantic Segmentation Notebook**
Tensorflow was used as the main modeling library for this notebook. Using transfer learning and a UNet architecture with additional pooling yielded the best results on the training dataset. I did also try a Resnet architecture with imagenet weights and found it yielded similar results. However the exported model size of a Resnet architecture was significantly larger.
 
A large amount of learning around image data processing and how to evaluate segmentation results went into this notebook. This notebook is where a large portion of my time was spent during this project.
 
![Image Processing](../../output/content/Data_process_1.jpg)
<br>
 
### **Data Processing**
One of the biggest learning lessons of this project was image processing to ensure the right shape to be ingested by our model. As satellite images are high resolution files we do not have powerful enough resources to run models over all pixels as a single input. To accommodate for this, a large image has to be broken down into smaller patches (sometimes referred to as image tiles). Once both images and training masks are divided into patches the data can then be inputted into the model for training.
 
![Process Detail](../../output/content/data_process_detail.jpg)
 
Once the model has been trained and patches predicted, the smaller image tiles need to be stitched back together to make an original image. This was done originally with manual code and then using the library of [Patchify](https://github.com/Vooban/Smoothly-Blend-Image-Patches) to assist in the process. A shoutout to [DigitalSreeni](https://www.youtube.com/c/DigitalSreeni) for fixing bugs in the library that has not been updated in the last few years.
 
I also experimented with using a smooth blending technique where tiles overlap and thus we can use a gaussian blend to smooth out edge artifacts. This is the approach I used in the final model as it picked up more features in images than without.
![Patched Smooth Blend](../../output/content/patch_smooth_2_3.jpg)
 
<br>
 
### **Model, Results & Application**
 
Using the Unet approach as my final model converged on a reasonable Jaccard Coefficient.
![UNet Results](../../output/content/Unet_results_2.jpg)
![Evan Metric](../../output/content/segment_result_eval.jpg)
 
Originally this model was trained on a dataset of Dubai with 6 labels. After tuning the model to satisfactory levels, I experimented with predicting segmentation masks on BC satellite images. Even though this model was trained on a different context (Dubai) than what its intended application was to be (logging in BC) it is a starting step to this continued experiment. To my suprise the model was able to pick out features such as clear cut forest quite well. The next steps would be to aquire training mask data for the intended region of use and compare results.
![BC Example](../../output/content/segment_result_bc.jpg)
![BC Example](../../output/content/segment_result_bc2.jpg)
 
I also created a streamlit script to host this model as a proof of concept. I used a slider to help users explore the results of their uploaded image.
![Streamlit Example](../../output/content/Streamlit%20Example.jpg)

