def mount_bucket(bucket_name):
  """
  This fn authenticates your google account and mounts to you Google Cloud Platform bucket
  """
  !echo "deb http://packages.cloud.google.com/apt gcsfuse-`lsb_release -c -s` main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
  !curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
  !sudo apt-get -y -q update
  !sudo apt-get -y -q install gcsfuse

  from google.colab import auth
  auth.authenticate_user()
  !mkdir -p gcp_bucket 
  !gcsfuse --implicit-dirs --limit-bytes-per-sec -1 --limit-ops-per-sec -1 {bucket_name} gcp_bucket

def bucket_data_transfer():
  """
  Unzips the jpeg dataset labels csv from our GCP bucket to the correct instance directory as staging for modeling.
  """
  # unzip from CGP bucket to instance
  !unzip /content/gcp_bucket/zip_files/semantic_walkthrough.zip -d {root_directory}
  !cp /content/gcp_bucket/imports/simple_multi_unet_model.py /content/
  !cp /content/gcp_bucket/imports/smooth_tiled_predictions.py /content/
  
def convert_hex(hex):
  """
  converts HEX string into array of RGB numbers
  """
  output = hex.lstrip('#')
  output = np.array(tuple(int(output[i:i+2], 16) for i in (0,2,4)))
  return output

def rgb_to_2d_label(label):
  """
  suppy mask lable as rgb
  converts pixels into label id (1,2,3 etc)
  """

  label_seg = np.zeros(label.shape, dtype=np.uint8)
  label_seg[np.all(label == Building, axis=-1)] = 0
  label_seg[np.all(label == Land, axis=-1)] = 1
  label_seg[np.all(label == Road, axis=-1)] = 2
  label_seg[np.all(label == Vegetation, axis=-1)] = 3
  label_seg[np.all(label == Water, axis=-1)] = 4
  label_seg[np.all(label == Unlabeled, axis=-1)] = 5

  # only take first channel, dont need the remaining
  label_seg = label_seg[:,:,0]
  
  return label_seg

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
