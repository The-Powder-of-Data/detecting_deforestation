import streamlit as st
import pandas as pd
from itertools import chain

# containers are horizontal sections
header = st.container()
datset = st.container()
modelInfo = st.container()

# some formatting 
st.markdown(
    """
    <style>
    .main{
    background-color:#184201
    }
    ,/STYLE
    """,
    unsafe_allow_html=True
)


with header:
    st.title('Welcome to my awesome project!')
    st.text('In this project I will... ')

with datset:
    st.header('Amazon Deforestation Dataset')
    st.text('I found this dataset on kaggle...link..')

    label_data = pd.read_csv('data/train_classes.csv')
    st.write(label_data.head())

    #show bar chart of categories
    st.subheader('Label Occurance Count')
    labels_list = list(chain.from_iterable([tags.split(" ") for tags in label_data['tags'].values]))
    labels_count = pd.Series(labels_list).value_counts()
    st.bar_chart(labels_count)

with modelInfo:
    st.header('Model Approach: Resnet50 Trasfer Learning')

    st.markdown('* **Our best result was using the Resne50 architecture with pretrained ImageNet weights**')
    st.markdown('* **We used pretrained weights from Imagenet as a starting point**')
    st.markdown('* **Then we trained with a discriminative learning rate on our dataset**')

    sel_col, disp_col = st.columns(2)

    fucked_level = sel_col.slider('How fucked is our plannet?', min_value=0,max_value=10, value=5, step=1)
    n_trees = sel_col.selectbox('How many trees are there is the world?', options=[100,200,300, 'No limit'], index=0)
    input_feature = sel_col.text_input('Which forest should we protect?', 'PULocationID')

    st.markdown('display test image')
    st.image('data/train_1.jpg')