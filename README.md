![Title Graphic](output/content/Title_image.png)
 
This project explores different methods to identify deforestation through satellite imagery data. The initial ground work for this repository was done as an individual 2 week project whilst studying Data Science at Lighthouse Labs. I am currently continuing to build out this repository.
<br>

<br>
 
# **Content**
1. [Overview](https://github.com/The-Powder-of-Data/detecting_deforestation#overview)
2. [Modeling](https://github.com/The-Powder-of-Data/detecting_deforestation#modeling)
4. [Cloud](https://github.com/The-Powder-of-Data/detecting_deforestation#cloud)
5. [Resources](https://github.com/The-Powder-of-Data/detecting_deforestation#resources)
 
<br>
 
# **Overview**
![Overview Image](output/content/Overview_image.png)
Welcome to my project on using satellite images to detect deforestation. I have a passion for sustainability and technology that enable us to make an impact. Applying a data science lens to earth monitoring is an area that I am striving to learn more about and thus the focus of this project.
 
Currently we are at an inflection point where we risk [only having 10% of the worlds rainforests](https://www.thebalance.com/deforestations-economic-impact-4163723) remaining by 2030. There is ***no Planet B*** and collectively we need to make more steps to make a measurable difference in the declining state of our world's ecological balance.
 
Through advancement in technology, such as with satellite imagery, there is more access to open source data and tools than ever before. As an individual you have access to free [open source satellite data](https://gisgeography.com/free-satellite-imagery-data-list/) that can be downloaded and leveraged today. One of the personal aims for this project was to better understand the landscape of modern tools for earth monitoring and practice creating my own suit of ML solutions.
 
<br>
 
## **The Vision**
My vision for this project was to use combine my Data Science learning and Satellite Imagery to build an ***Analytics Time Machine***. For my initial efforts I focused on the first 2 components.
![Project Vision](output/content/vision_1.PNG)
 
<br>
 
## **Planning & Presentation**
I am a very visual worker. My life and brain exists on the platform Miro which is the best online whiteboard I have found so far. You may be amused to hear that I have successfully workshopped virtual dates whilst in a long distance relationship on Miro! If interested I have linked my Miro workspace, final presentation and a few resources. If you ever want to talk about visual learning and collaboration platforms please feel free to reach out to me on [LinkedIn](https://www.linkedin.com/in/josephluiz/). 

- [Miro Workspace](https://miro.com/app/board/uXjVO2axXTY=/?share_link_id=285273823775): This is where my visualy capture my brain, excuse the mess
- [Presentation Recording](): If you would like to watch the 5 minute presentation (coming soon)
- [Project Gannt](https://miro.com/app/board/uXjVO1LxdYg=/?share_link_id=321967812507): Done in Miro and shared with my cohort to help them manage their delivery timeline

# **Modeling**
## **Multi Label Classification**
I started with the first layer to my solution. Learning how to approach a supervised multi label classification model. These results can then be used as layer 1 in our overall understanding and automation of processing satellite imagery of our earth.

![MLC Example](output/content/MLC_example_1.png)
 
 
### **Data & Processing**
As a start point I used the [Planet dataset](https://www.kaggle.com/competitions/planet-understanding-the-amazon-from-space/data). The original 22gb torrent is no longer being seeded, however there is an [alternative link](https://www.kaggle.com/datasets/nikitarom/planets-dataset) to a balanced sample that you can download. <br>
For this example dataset you could choose not to do any pre-processing as the satellite images come already broken into small enough tiles for preprocessing. If ingesting images from a [satellite API](https://www.programmableweb.com/news/top-10-satellites-apis/brief/2020/06/14) you would have to preprocess images into small enough patches/tiles in order for them to be ingested by the model. To see an example of this please look at the Semantic Segmentation Model approach.
 
<br>
 
### **Results & Application**
FastAI was a great library once learning the basics. Using a I was able to get strong results using a few techniques.
- Transfer learning on a Resnet50 architecture and imagenet weights
- Discriminate Learning Rates
- Image sizing trick to increase the size of our dataset
<br>
After a variety of iterations the max result obtained was an f2 of 0.93 which puts this model in the top 60 results of 888 entries in the original competition.
![MLC Results](output/content/MLC_result.jpg)
<br>
 
## **Semantic Segmentation**
The second layer to the envisioned tool was to have pixel by pixel identification on each image.
For this Tensorflow was used as the main library. Using transfer learning and a UNet architecture with additional pooling yielded the best results on the training dataset. I did also try a Resnet architecture with imagenet weights and found it yielded similar results. However the exported model size of a Resnet architecture was significantly larger.
A large amount of learning around image data processing and how to evaluate segmentation results went into this notebook. This notebook is where a large portion of my time was spent during this project.
![Image Processing](output/content/Data_process_1.jpg)
<br>
 
### **Data Processing**
One of the biggest learning lessons of this project was image processing to ensure the right shape to be ingested by our model. As satellite images are high resolution files we do not have powerful enough resources to run models over all pixels as a single input. To accommodate for this, a large image has to be broken down into smaller patches (sometimes referred to as image tiles). Once both images and training masks are divided into patches the data can then be inputted into the model for training.
![Process Detail](output/content/data_process_detail.jpg)
Once the model has been trained and patches predicted, the smaller image tiles need to be stitched back together to make an original image. This was done originally with manual code and then using the library of [Patchify](https://github.com/Vooban/Smoothly-Blend-Image-Patches) to assist in the process. A shoutout to [DigitalSreeni](https://www.youtube.com/c/DigitalSreeni) for fixing bugs in the library that has not been updated in the last few years.
I also experimented with using a smooth blending technique where tiles overlap and thus we can use a gaussian blend to smooth out edge artifacts. This is the approach I used in the final model as it picked up more features in images than without.
![Patched Smooth Blend](output/content/patch_smooth_2_3.jpg)
<br>
### **Model, Results & Application**
Using the Unet approach as my final model converged on a reasonable Jaccard Coefficient (similar to Mean IOU) of 0.73.
![UNet Results](output/content/Unet_param_3.jpg)
![Evan Metric](output/content/segment_result_eval.jpg)
Originally this model was trained on a [dataset](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery) of Dubai with 6 labels. After tuning the model to satisfactory levels, I experimented with predicting segmentation masks on BC satellite images. Even though this model was trained in a different context (Dubai) than what its intended application was to be (logging in BC) it is a starting step to this continued experiment. To my suprise the model was able to pick out features such as clear cut forest quite well. The next steps would be to aquire training mask data for the intended region of use and compare results.
![BC Example](output/content/segment_result_bc.jpg)
![BC Example](output/content/segment_result_bc2.jpg)
 
<br>
 
# **Cloud**
 
## **Google Cloud Platform**
In this project I focused on using *Google Cloud Platform* as much as possible. As I had previous experience with AWS I wanted to take the opportunity to learn what GCP had to offer. In the spirit of everything Google I also used Google Colab as my main IDE to utilize access to a GPU for training.
 
For files included in this repo I have authenticated through my personal email with the project Google Cloud Buckets. If you would like to replicate this approach please setup a GCP account and you will get $300 for free for 80 days. Alternatively you could host and reference your files locally.
 
Setting this product up in the cloud is still a work in progress but below are some of the key areas I experimented with and learnt more about.
 
- Managing a storage bucket for image and model storage
- BigQuery for table to pull and push data from
- Compute Engine VM as an instance to test hosting of an api
- App Engine to host a containerized version of our Streamlit app
- Vertex AI for model and maintenance 
 
As I found my data processing time slower than ideal when using the base Streamlit cloud service, I decided I wanted to setup an API in GCP that I could scale as needed to increase the output time of model predictions. Below is my original concept diagram of how a streamlit script would flow with GCP Bucket through an api to deliver quicker results.
![API Diagram](output/content/API%20Architecture.jpg)
 
## **Streamlit Dashboard**
As I learn more about Streamlit I was able to get an MVP ready within the short timeline of this project. I did have some hurdles successfully hosting this solution in the cloud. After my Data Science bootcamp I will be back to revisit this learning architecture. My aim is to have a flywheel setup so the model is hosted, can take in new user data and learn in cycles.
 
<br>
 
As a result of my explorations I created a streamlit script to host this model as a proof of concept. I used a slider to help users explore the results of their uploaded image.
![Streamlit Example](output/content/streamlit_eg_2.jpg)
 
# Resources
 
### General Research
- [Global Forest Watch](https://www.globalforestwatch.org/)
- [Using Satellite Imagery to Investigate Deforestation on GFW](https://www.youtube.com/watch?v=bhm7dgqfsw4)
- [Planet NICFI Platform](https://www.planet.com/nicfi/)
- [Global Forest Change Dashboard](https://glad.earthengine.app/view/global-forest-change#dl=1;old=off;bl=off;lon=-123.99617081539049;lat=49.45589555325505;zoom=8;)
- [SentinelHub Browser](https://apps.sentinel-hub.com/eo-browser/)
- [BC Map Services](https://www2.gov.bc.ca/gov/content/data/geographic-data-services/web-based-mapping/map-services)
- [Deforestion Economic Impact](https://www.thebalance.com/deforestations-economic-impact-4163723#citation-13)
- [Satellite Imagery Market Info](https://www.alliedmarketresearch.com/commercial-satellite-imaging-market#:~:text=Commercial%20Satellite%20Imaging%20Market%20Statistics,industry%20on%20a%20global%20level.)
- [Largest Satellite Imagery Companies](https://www.appgeo.com/top-satellite-and-aerial-imagery-companies/)
- [Stats Can Satellite Image Story](https://www.statcan.gc.ca/en/data-science/network/satellite-imaging)
- [Satellite Image Data Consumption](https://datacenterfrontier.com/terabytes-from-space-satellite-imaging-is-filling-data-centers/)
 
### Achademic Papers / Articles
- [A machine learning approach to map tropical selective logging](https://www.sciencedirect.com/science/article/pii/S0034425718305534)
- [Integrated Segmentation Approach with Machine Learning Classifier in Detecting and Mapping Post Selective Logging Impacts Using UAV Imagery](https://www.mdpi.com/1999-4907/13/1/48/htm)
- [Detecting deforestation from satellite images](https://towardsdatascience.com/detecting-deforestation-from-satellite-images-7aa6dfbd9f61)
- [Leveraging satellite imagery for machine learning computer vision applications](https://medium.com/artefact-engineering-and-data-science/leveraging-satellite-imagery-for-machine-learning-computer-vision-applications-d22143f72d94)
- [Fire-Net: A Deep Learning Framework for Active Forest Fire Detection](https://www.hindawi.com/journals/js/2022/8044390/)
- [Active Fire Detection Using a Novel Convolutional Neural Network Based on Himawari-8 Satellite Images](https://www.frontiersin.org/articles/10.3389/fenvs.2022.794028/full)
 
### Learnings
- [Digital Sreeni Walkthrough on Semantic Segmentation](https://www.youtube.com/c/DigitalSreeni)
- [Jeremy Howard Fastai Walkthrough](https://www.youtube.com/watch?v=MpZxV6DVsmM)
- [Satellite-image-deep-learning](https://github.com/robmarkcole/satellite-image-deep-learning)
- [Streamlit Walkthrough](https://www.youtube.com/watch?v=vIQQR_yq-8I)
- [Smooth Blend Patches](https://github.com/Vooban/Smoothly-Blend-Image-Patches)
- [GCP Flask Setup](https://www.section.io/engineering-education/deploy-flask-to-gce/)
- [Deploy a ML Flywheel](https://www.youtube.com/watch?v=fw6NMQrYc6w)
- [Planet API Example](https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/data-api-tutorials/search_and_download_quickstart.ipynb)
 
### Platforms
- [Streamlit](https://streamlit.io/)
- [Google Colab](https://colab.research.google.com/?utm_source=scs-index)
- [Google Cloud Platform](console.cloud.google.com/home)
 

