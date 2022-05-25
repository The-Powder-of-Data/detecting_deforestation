import streamlit as st
import os
import pandas as pd
import numpy as np
from itertools import chain
import fn 
from itertools import chain
import matplotlib.pyplot as plt
from PIL import Image 

from google.cloud import storage
from google.oauth2 import service_account
#from google.cloud import bigquery
#import pandas_gbq

from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
import cv2
from sklearn.preprocessing import MinMaxScaler
from module.imports_smooth_tiled_predictions import predict_img_with_smooth_windowing
from tensorflow.keras.models import load_model

from streamlit_image_comparison import image_comparison # testing slider

# set page config
#st.set_page_config(page_title="Image-Comparison Example", layout="centered")


########################
#fns
@st.cache
def load_image(image_file):
    # need to make sure it converts to BGR to use in model
    img = Image.open(image_file).convert('RGB')
    return img


def label_to_rgb(predicted_image):
    """
    converts predicted mask into label rgb colors
    """
    Building = '#3C1098'.lstrip('#')
    Building = np.array(tuple(int(Building[i:i+2], 16) for i in (0, 2, 4))) # 60, 16, 152
    
    Land = '#8429F6'.lstrip('#')
    Land = np.array(tuple(int(Land[i:i+2], 16) for i in (0, 2, 4))) #132, 41, 246
    
    Road = '#6EC1E4'.lstrip('#') 
    Road = np.array(tuple(int(Road[i:i+2], 16) for i in (0, 2, 4))) #110, 193, 228
    
    Vegetation =  'FEDD3A'.lstrip('#') 
    Vegetation = np.array(tuple(int(Vegetation[i:i+2], 16) for i in (0, 2, 4))) #254, 221, 58
    
    Water = 'E2A929'.lstrip('#') 
    Water = np.array(tuple(int(Water[i:i+2], 16) for i in (0, 2, 4))) #226, 169, 41
    
    Unlabeled = '#9B9B9B'.lstrip('#') 
    Unlabeled = np.array(tuple(int(Unlabeled[i:i+2], 16) for i in (0, 2, 4))) #155, 155, 155
    
    segmented_img = np.empty((predicted_image.shape[0], predicted_image.shape[1], 3))
    
    segmented_img[(predicted_image == 0)] = Building
    segmented_img[(predicted_image == 1)] = Land
    segmented_img[(predicted_image == 2)] = Road
    segmented_img[(predicted_image == 3)] = Vegetation
    segmented_img[(predicted_image == 4)] = Water
    segmented_img[(predicted_image == 5)] = Unlabeled
    
    segmented_img = segmented_img.astype(np.uint8)
    return(segmented_img)

@st.cache
def pipeline_pred_smooth(img_file):
    """
    Takes an image directory and produces a segmentation mask using a smoothed edge tile approach

    0. takes image
    1. image scaled
    2. break image into tiles w overlapping pixels
    3. use model to predict segemntation on each tile
    4. convert array back to rgb
    5. display
    """
    #img = cv2.imread(img_file)
    # or do i just use... 
    img = img_file 
    scaler = MinMaxScaler()
    patch_size = 256
    n_classes = 6

    model = load_model('data/models_segment_unet_100e.hdf5', compile=False)
    input_img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)

    predictions_smooth = predict_img_with_smooth_windowing(
        input_img,
        window_size=patch_size,
        subdivisions=2,  
        nb_classes=n_classes,
        pred_func=(
            lambda img_batch_subdiv: model.predict((img_batch_subdiv))))

    final_prediction = np.argmax(predictions_smooth, axis=2)
    prediction_w_smooth = label_to_rgb(final_prediction)
    return prediction_w_smooth


########################

# Google BigQuerry Authenticate

# Create API client.
# credentials = service_account.Credentials.from_service_account_info(
#     st.secrets["gcp_service_account"]
#     )

# client = bigquery.Client(credentials=credentials)


########################

#sidebar stuff
nav = st.sidebar.radio("Navigation", ['Home', "Multi Label Classification", "Semantic Segmentation"])

if nav == 'Home':
    st.write('About us page')

# containers are horizontal sections
header = st.container()
imageup = st.container()
button = st.container()
results = st.container()
image_frontend = st.container()
feedback = st.container()
goog_test = st.container()


with header:
    st.title('Deforestation Detection')
    st.text('Upload an image for a prediciton mask')

with imageup:
    # st.title('Upload Image')
    uploaded_image = st.file_uploader(' ', type=['png', 'jpeg', 'jpg'])
    

    if uploaded_image is not None:
        file_details = {"FileName":uploaded_image.name,"FileType":uploaded_image.type,"FileSize":uploaded_image.size}
        image_data = uploaded_image.getvalue() # do we putthis in the pred fn?
        st.write(file_details) 
        #st.text("thank you for the image... your prediction is being processed")
        
        img = load_image(uploaded_image) # this adds an extra dimension vs cv2
        #img = cv2.imread(uploaded_image)
        img_frame = np.array(img)
        #st.write(img_frame)

    else:
        pass
        #st.text('when ready please upload an image')

prediction = None # setting this variable so we can use the if statement in results

with button:
    if st.button('Analyize'):
        prediction = pipeline_pred_smooth(img_frame)
        #st.write(prediction)
    else:
        pass


with results:
    st.title('Display Results')
    
    col1, col2 = st.columns(2)
    
    if prediction is not None:
        with col1:
            st.write('Original Image')
            st.image(img)

        with col2:
            st.write('Segmented Prediction')
            st.image(prediction)

        # fig, ax = plt.subplots(figsize=(16,16))
        # ax.imshow(img)
        # ax.axis('off')
        # st.pyplot(fig)

    else:
        st.text('Please upload image above for a prediction')

# testing slider


with image_frontend:
    if prediction is not None:
        
        image_comparison(
            img1=img,
            img2=prediction,
            label1="Original Image",
            label2="Segmented Prediction",
            show_labels=False,
            in_memory=True
        )
    else:
        pass


with feedback:
    st.title('User Feedback')
    feedback = st.text_input('Input User Feedback :D')

### Google big query test

# @st.experimental_memo(ttl=600) # this refreshs table every 10 min
# def run_query(query):
#     query_job = client.query(query)
#     rows_raw = query_job.result()
#     # Convert to list of dicts. Required for st.experimental_memo to hash the return value.
#     rows = [dict(row) for row in rows_raw]
#     return rows

########################
# #google test
# sql = "SELECT * FROM `august-creek-349822.labels.amazon_mlc_labels` ORDER BY string_field_0 LIMIT 10"
# project_id = "august-creek-349822"

# df = pandas_gbq.read_gbq(sql, project_id=project_id)

# with goog_test:
#     st.dataframe(df)
