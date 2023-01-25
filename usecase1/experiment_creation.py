import pandas as pd
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
#warnings.simplefilter('ignore', category=VerifyWarning)
import numpy as np
import os

import json

SAVE_IMG = True

EXP_NAME = "croppedbackyard"

EXP_STRUCTURE = {
        "data": {
            "type": "folder",
            "content": "list of npy"
        },
        "metadata": {
            "type": "file",
            "name": "metadata.yml"
        },
        "info": {
            "type": "file",
            "name": "info.json"
        }
}

if not os.path.exists(EXP_NAME):
    os.makedirs(EXP_NAME)

data_folder = os.path.join(EXP_NAME,"data")
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

df = pd.read_json('data/datalists/datalist.json')

info_json = df["data"]

source_name_list =  map(lambda x: x["sname"], info_json)
survey_list =       map(lambda x: x["survey"], info_json)
label_list =        map(lambda x: x["label"], info_json)
telescope_list =    map(lambda x: x["telescope"], info_json)
project_list =      map(lambda x: x["project"], info_json)
path_list =         map(lambda x: x["filepaths"][0].removeprefix("/storage/riggi/Data/MLData/smorph-dataset/"), info_json)
target_path_list =  map(lambda i: "data/{}-{}-{}.npy".format(i, info_json[i]["sname"], info_json[i]["survey"]), info_json.index)
#source_key =  map(lambda x: "{}/{}-{}.npy".format(exp_name, x["sname"], x["survey"]), info_json)

info_pd = pd.DataFrame()
info_pd["source_name"] =   list(source_name_list)
info_pd["source_type"] =   list(label_list)
info_pd["survey"] =        list(survey_list)
info_pd["telescope"] =     list(telescope_list)
info_pd["project"] =       list(project_list)
info_pd["original_path"] = list(path_list)
info_pd["target_path"] =   list(target_path_list)

max_shape_list = []
min_shape_list = []
ratio_list = []

for idx, record in info_pd.iterrows():
    #print(record["source_name"])
    hdul = fits.open(record["original_path"])
    #print(hdul[0].header[''])
    npy_image = hdul[0].data
    #np.save("raw_dataset/{}-{}.npy".format(record["source_name"], record["survey"]), npy_image)
    
    """
    Preprocessing
    """
    #min_image = np.min(npy_image)
    #npy_image[npy_image == 0.0] = min_image
    #np.save("preprocessed_dataset/{}-{}.npy".format(record["source_name"], record["survey"]), npy_image)
    #max_image = np.max(npy_image)
    #npy_image = (npy_image - min_image) / (max_image - min_image) # MinMax normalization
    min_image = np.min(npy_image)
    max_image = np.max(npy_image)
    max_shape = np.max(npy_image.shape)
    min_shape = np.min(npy_image.shape)
    #if (min_image < 0.) or (max_image > 1.) or (min_image == max_image) or (np.any(np.isnan(npy_image))):
    if (max_shape < 8) or (min_image == max_image) or (np.any(np.isnan(npy_image))):
        print(max_shape)
        #print("min = max")
        #print(min_image == max_image)
        #print("NAN")
        #print(np.any(np.isnan(npy_image)))
        #print(record["original_path"])
        #print(record["source_name"])
        #print(record["survey"])
        print(idx)
        info_pd = info_pd.drop(index=(idx))
        #print(info_pd.describe())
    else:
        if record["source_type"] == "BACKGROUND":
            min_crop_size = 8
            min_x = np.random.randint(0, npy_image.shape[0] - min_crop_size*2, size=1)[0]
            min_y = np.random.randint(0, npy_image.shape[1] - min_crop_size*2, size=1)[0]
            max_x = np.random.randint(min_x+min_crop_size, npy_image.shape[0], size=1)[0]
            max_y = np.random.randint(min_y+min_crop_size, npy_image.shape[1], size=1)[0]
            npy_image = npy_image[min_x:max_x, min_y:max_y]
        max_shape = np.max(npy_image.shape)
        min_shape = np.min(npy_image.shape)
        ratio = min_shape / max_shape
        max_shape_list.append(max_shape)
        min_shape_list.append(min_shape)
        ratio_list.append(ratio)
        if SAVE_IMG:
            np.save(os.path.join(EXP_NAME,record["target_path"]), np.expand_dims(npy_image, -1))

info_pd = info_pd.reset_index(drop=True)
print(info_pd.describe())
info_pd["max_shape"] = max_shape_list
info_pd["min_shape"] = min_shape_list
info_pd["ratio"] = ratio_list
result = info_pd.to_json(orient="index")
parsed = json.loads(result)
print(info_pd.describe())

#print(json.dumps(parsed, indent=4))
out_file = open(os.path.join(EXP_NAME, "info.json"), "w+")
json.dump(parsed, out_file, indent=4)