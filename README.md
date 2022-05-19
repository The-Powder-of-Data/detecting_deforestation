# Detecting Deforestation
This project explores different methods to identify deforestation from satellite imagery. I did the initial ground work for this repository as my final project whilst studying Data Science. I am now continuing to build upon it to develop my skills.
<br>
*Dashboard Link*
*PPT Link*

## Content
1. Overview
2. Modeling & Data
4. Cloud Hosting
5. Resources

## Overview

## Modeling & Data 
- **Multi Label Classification Model**: 
We started by applying a multi labeling approach to an image dataset. The dataset used was the *planets_dataset* aready preprocessed and labeled. The best results generated were using transfer learning on a *resnet50* architecture with *imagenet* weights. Techniques such as discriminative learning rates and image resizing were used to optimize our training results with a maximum fbeta of 0.93. This is a very strong result relative to the scoreboards for this dataset. It was quite remarkable to explore how combining a few simple techniques could create such a robust model in this example.  

---- An evaluation metric of f2 was chosen because.... 
---- Images

- **Semantic Segmentation Model**: 
The next approach was pixel by pixel classification, otherwise known as semantic segmentation. The biggest challenge for this was finding a relevent dataset with human created masks to be used for supervised learning. We ended up using a classic dataset containing arial images of Dubai with 6 lables and human created masks for each image. This dataset required pre-processing as images were too large to feed into a model for training and prediction. I created a pipeline where images are broken into smaller patches along with their coresponding masks. I shoutout to *link* for a clear breakdown of different ways to approach this. 

When the model was finalised we had a jacard_coeficient (similar to Mean IOU) of 0.752 for our validation set. We used *tensorflow* to handle training of a *UNet* architecture with some additional pooling and *imagenet* weights to start. The model ran for about 100 epochs.  

Once our model was trained and predictions made a process was created to rejoin our previously processed patches. We do want a full size origional image after all. In this process it was found that is we simply stich each patch together there can be edge artifacts. So a blended approach was taken where each patch had a slight overlap with patches around it. Thus, allowing for a gausian blended apporach to be used and smooth out any edges. 

---- Image of results
*check out some results in our dashboard*

## Cloud Hosting
In this project I focused on using *Google Cloud Platform* to learn more about their cloud offering. <br>
- A storage bucket was spun up for image and model storage
- BigQuery for table to pull and push data from
- Compute Engine VM as an instance to test our api
- App Engine to host a containerized version of our streamlit app

## Resources
