import os
import cv2
import sys
import math
import scipy
import random
import rasterio
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import seed
from glob import glob
from rasterio.windows import Window

from keras.models import model_from_json
from keras import backend as K
from keras.layers import Conv2D
from keras import layers
from keras.models import Model
import tensorflow as tf
import random
import json
import copy

from utils.augmentation import augment 

class Generator:
    def __init__(self, batch_size, class_0, class_1, class_2, class_3, class_4, num_channels):
        self.num_channels = num_channels
        self.num_classes = 4
        self.IMG_ROW = 1024
        self.IMG_COL = 1024
        self.batch_size = batch_size

        self.class_0 = class_0
        # self.class_1 = class_1
        # self.class_2 = class_2
        # self.class_3 = class_3
        # self.class_4 = class_4

        # self.cloud = False
        self.augm = False
        self.color_aug_prob = 0
        self.stands_id = False
        # self.per_stand_loss = False
        self.instance_augm = False
        self.instance_augm_prob = 0.6
        self.background_prob = 0.6
        self.shadows = False
        self.extra_objects = False
        
        # self.prob_split = False
        
        self.channels = ['pre_r', 'pre_g', 'pre_b']
        # self.channels_background = ['RED', 'GRN', 'BLU']
        self.channels_background = ['Aerosols', 'Blue', 'Green', 'Red', 'Red Edge 1', 'Red Edge 2', 'Red Edge 3', 'NIR', 'Red Edge 4', 'Water vapor', 'SWIR1', 'SWIR2']
        self.val = False
        # self.background_list_val = []
        # self.background_list_train = []
        
        # deler opp bildet, dette trenger vel ikke jeg?
        self.val_upper_threshold = 0
        self.val_lower_threshold = 1620
        self.train_upper_threshold = 1620
        self.train_lower_threshold = 4418 
            
    def get_img_mask_array(self, imgpath, upper_left_x, upper_left_y, pol_width, pol_height, crop_id, age_flag = False):
        with rasterio.open(imgpath+'/'+ self.channels[0] + '.tif') as src:
            size_x = src.width # lagrer høyde og bredde
            size_y = src.height
        # forskjell mellom dimensionen til bildet og polygonet (non-negative)
        # skjønner ikke hvorfor
        difference_x = max(0, self.IMG_COL - int(pol_width))
        difference_y = max(0, self.IMG_ROW - int(pol_height))
        
        # randomiserer x og y for å få et tilfeldig utsnitt av bildet?
        rnd_x = random.randint(max(0, int(upper_left_x) - difference_x),
                               min(size_x, int(upper_left_x) + int(pol_width) + difference_x) - self.IMG_COL)
        rnd_y = random.randint(max(0, int(upper_left_y) - difference_y),
                               min(size_y, int(upper_left_y) + int(pol_height) + difference_y) - self.IMG_ROW)
        # subset av bildet som kan brukes til å plassere objekter?
        window = Window(rnd_x, rnd_y, self.IMG_COL, self.IMG_ROW)
        
        # initializes 3D array med 0er
        # IMG_ROW x IMG_COL er størrelsen til bildet? 
        # 1 x 1024 x 1024?
        mask_0 = np.zeros((1, self.IMG_ROW, self.IMG_COL))

        # itererer over klassene i class_0 og summerer de (?)
        # window=window fordi det skal være innenfor et område?
        for cl_name in self.class_0:
            with rasterio.open(imgpath + '/{}.tif'.format(cl_name)) as src:
                mask_0 += src.read(window=window).astype(np.uint8)
        
        # hente ut et nytt objekt?
        # lager en binary maske for objektene i et bilde?
        if self.stands_id:
            # blir brukt til å hente ut masker til objekter
            with rasterio.open(imgpath + '/id_mask.tif') as src:
                mask_id = src.read(window=window).astype(np.uint16)
                mask_id = np.where(mask_id[0,:,:]==int(crop_id), 1, 0) 
        
        #tiff.imshow(mask_id)
        #tiff.imshow(mask_0)

        # 3D array med 1024 x 1024 x 12 for meg?
        img = np.ones((self.IMG_ROW, self.IMG_COL, self.num_channels), dtype=np.uint8)
        # for alle kanaler (12?) åpne bilde og lagre det i img
        for i, ch in enumerate(self.channels): 
            with rasterio.open(imgpath+'/'+ch+ '.tif') as src:
                img[:,:,i] = src.read(window=window)
        # output: img.shape = (1024, 1024, 12) for meg
        
        # 1024 x 1024 x 5
        mask = np.ones((self.IMG_ROW, self.IMG_COL, self.num_classes))
        
        # masken for første channel
        mask[:,:,0] = mask_0 
        
        # sette bilder til å bli augmentert?
        flag_instance_augm = False
        
        # augmenter med en viss sannsynlighet om instrance_augm = True
        if self.instance_augm and random.random() < self.instance_augm_prob: #self.val==False
            
            #---------------------------------------------------------------------------------------
            # crop target object
            #---------------------------------------------------------------------------------------           
            mask[:,:,0] = mask_0 * mask_id
            # henter ut objektet (masken) i den første kanalen
            
            # 1024 x 1024 x 3, burder være 1024 x 1024 x 12
            mask_tmp = np.ones((self.IMG_ROW, self.IMG_ROW, 3))
            # for hver maske, om verdier er større enn 0.5, sett til 1
            # kan skyldes at de endrer på størrelsen og får en verdi mellom 0 og 1, ikke en av de
            mask_tmp[:,:,0] = mask[:, :, 0]>0.5
            mask_tmp[:,:,1] = mask[:, :, 0]>0.5
            mask_tmp[:,:,2] = mask[:, :, 0]>0.5
            
            # crop building
            building_mask = np.ones((self.IMG_ROW, self.IMG_ROW, 3))
            # drar ut verdiene til masken fra det originale bildet
            building_mask[:,:,0] = img[:, :,0] * mask_id
            building_mask[:,:,1] = img[:, :,1] * mask_id
            building_mask[:,:,2] = img[:, :,2] * mask_id
            
            # roterer objekt på egenhånd? Men dette skjer vel i augmenteringen?
            #angle = 0# random.randint(0, 180)
            #new_img = rotate(building_mask, angle)
            
            #rows, colomns, _ = np.where(new_img>0)
            
            #rnd_x = 0 # random.randint(max(0, rows[0] - (128-rows[-1]+rows[0])),
                               #min(new_img.shape[0], rows[0] + 128) - 128)

            #rnd_y = 0 #random.randint(max(0, colomns[0] - (128-colomns[-1]+colomns[0])),
                              # min(new_img.shape[1], colomns[0] + 128) - 128)
            #img = new_img[rnd_x:rnd_x+self.IMG_ROW,rnd_y:rnd_y+self.IMG_ROW,:]
            
            #mask[:,:,0] = rotate(mask_tmp.astype(np.uint8), angle)[rnd_x:rnd_x+self.IMG_ROW,rnd_y:rnd_y+self.IMG_ROW,0]
            
            # add extra objects
            if self.extra_objects:
                # for alle objekter 
                for _ in range(random.randint(0, self.extra_objects)):
                    # velg et tilfedlig objekt fra class_0 (bygninger)
                    random_key = random.choice(list(self.json_file_cl0_train.keys()))
                    # henter ut bojektet/masken?
                    upper_left_x, upper_left_y, pol_width, pol_height = self.extract_val(self.json_file_cl0_train[random_key])

                    # henter ut navnet på bildet? objektet?
                    imgpath = random_key[:-len(random_key.split('_')[-1])-1]
                    crop_id = random_key.split('_')[-1]
                    #print('imgpath ', imgpath)
                    #print('crop_id ', crop_id)

                    # henter ut størrelsen på det valgte bildet
                    with rasterio.open(imgpath+'/'+ self.channels[0] + '.tif') as src:
                        size_x = src.width
                        size_y = src.height
                        
                    difference_x = max(0, self.IMG_COL - int(pol_width))
                    difference_y = max(0, self.IMG_ROW - int(pol_height))
                    
                    intersect_initial = True
                    search_iter = 0
                    while intersect_initial and search_iter < 10:
                        #print(search_iter)
                        search_iter += 1
                        #rnd_x = random.randint(max(0, int(upper_left_x) - difference_x),
                        #                       min(size_x, int(upper_left_x) + int(pol_width) + difference_x) - self.IMG_COL)
                        #rnd_y = random.randint(max(0, int(upper_left_y) - difference_y),
                        #                       min(size_y, int(upper_left_y) + int(pol_height) + difference_y) - self.IMG_ROW)
                        
                        rnd_x = random.randint(0, size_x - self.IMG_COL)
                        rnd_y = random.randint(0, size_y - self.IMG_ROW)


                        window = Window(rnd_x, rnd_y, self.IMG_COL, self.IMG_ROW)

                        mask_0 = np.zeros((1, self.IMG_ROW, self.IMG_COL))
                        for cl_name in self.class_0:
                            with rasterio.open(imgpath + '/{}.tif'.format(cl_name)) as src:
                                mask_0 += src.read(window=window).astype(np.uint8)

                        with rasterio.open(imgpath + '/id_mask.tif') as src:
                            mask_id = src.read(window=window).astype(np.uint16)
                            mask_id = np.where(mask_id[0,:,:]==int(crop_id), 1, 0) 
                        
                        # check if there is an intersection with the initial object
                        if np.sum(mask_id * mask[:,:,0]) == 0:
                            # no intersection
                            mask[:,:,0] += mask_id
                            intersect_initial = False
                            img_extra = np.ones((self.IMG_ROW, self.IMG_COL, self.num_channels), dtype=np.uint8)
                            
                            for i, ch in enumerate(self.channels):
                                with rasterio.open(imgpath+'/'+ch+ '.tif') as src:
                                    img_extra[:,:,i] = src.read(window=window)
                            
                            # building_mask: target object. Update med nye objektet? Bildet, ikke maske
                            building_mask[:,:,0] = img_extra[:, :,0] * mask_id + building_mask[:,:,0] * np.where(mask_id, 0, 1)
                            building_mask[:,:,1] = img_extra[:, :,1] * mask_id + building_mask[:,:,1] * np.where(mask_id, 0, 1)
                            building_mask[:,:,2] = img_extra[:, :,2] * mask_id + building_mask[:,:,2] * np.where(mask_id, 0, 1)

            # add augm for object crop
            # augment together
            if self.augm:
                building_mask, mask_tmp = augment(building_mask.astype(np.uint8), mask, self.color_aug_prob)
                # Oppdaterer masken med nye augmenterte objektet
                if len(mask_tmp.shape) == 2:
                    # hvis den kun er 2D, altså ikke flere channels
                    mask[:,:,0]=mask_tmp
                else:
                    # hvis augmentation resulterer i flere channels??
                    mask=mask_tmp

            # er dette nødvendig?
            building_mask = building_mask.astype(np.uint8)

            #---------------------------------------------------------------------------------------
            # find background
            #---------------------------------------------------------------------------------------
            
            # om bakgrunner skal bli augmentert?
            flag_background = False

            # om vi har bakgrunner, og gitt en sannsynlighet
            # with some probability, background should be augmented
            if len(self.background_list_train) and random.random() < self.background_prob:
                # sette om vi trener eller evaluerer
                if self.val:
                    imgpath = random.choice(self.background_list_val)
                else:
                    imgpath = random.choice(self.background_list_train) 
                # ja augmenter bakgrunnen
                flag_background = True
            else:
                # if 
                attempt = 0
                if self.val:
                    random_key = random.choice(list(self.json_file_cl0_val.keys()))
                else:
                    random_key = random.choice(list(self.json_file_cl0_train.keys()))
                imgpath = random_key[:-len(random_key.split('_')[-1])-1]

                with rasterio.open(imgpath+'/'+ self.channels[0] + '.tif') as src:
                    size_x = src.width
                    size_y = src.height

                rnd_x = random.randint(0, size_x-self.IMG_ROW-1)
                rnd_y = random.randint(0, size_y-self.IMG_ROW-1)
                window = Window(rnd_x, rnd_y, self.IMG_COL, self.IMG_ROW)
                mask_background = np.zeros((1, self.IMG_ROW, self.IMG_COL))
                with rasterio.open(imgpath + '/{}.tif'.format(cl_name)) as src:
                    mask_background += src.read(window=window).astype(np.uint8)

                # if background is not emtpy, altså det er et objekt eller?
                while np.sum(mask_background) > 0:
                    # if there are objects, try to find coordinates not overlapping?
                    attempt += 1
                    rnd_x = random.randint(0, size_x-self.IMG_ROW-1)
                    rnd_y = random.randint(0, size_y-self.IMG_ROW-1)
                    window = Window(rnd_x, rnd_y, self.IMG_COL, self.IMG_ROW)
                    mask_background = np.zeros((1, self.IMG_ROW, self.IMG_COL))
                    with rasterio.open(imgpath + '/{}.tif'.format(cl_name)) as src:
                        mask_background += src.read(window=window).astype(np.uint8)

                    # finne om den kan bli brukt eller ikke?
                    if attempt > 50:
                        # hvis det ikke er mulig, velg et nytt bilde
                        attempt = 0
                        if self.val:
                            random_key = random.choice(list(self.json_file_cl0_val.keys()))
                        else:
                            random_key = random.choice(list(self.json_file_cl0_train.keys()))
                        imgpath = random_key[:-len(random_key.split('_')[-1])-1]
                        with rasterio.open(imgpath+'/'+ self.channels[0] + '.tif') as src:
                            size_x = src.width
                            size_y = src.height


            background = np.ones((self.IMG_ROW, self.IMG_COL, self.num_channels), dtype=np.uint8)
            # if background can be used?
            if flag_background:
                # new background, not initial?
                channels_list = self.channels_background
                with rasterio.open(imgpath+'/'+ self.channels_background[0] + '.tif') as src:
                    size_x = src.width
                    size_y = src.height

                rnd_x = random.randint(0, size_x-self.IMG_ROW-1)
                rnd_y = random.randint(0, size_y-self.IMG_ROW-1)
                window = Window(rnd_x, rnd_y, self.IMG_COL, self.IMG_ROW)
            else:
                channels_list = self.channels

            for i, ch in enumerate(channels_list):
                with rasterio.open(imgpath+'/'+ch+ '.tif') as src:
                    background[:,:,i] = src.read(window=window)
            
            # add augm for background
            if self.augm:
                background, mask_tmp  = augment(background, mask, self.color_aug_prob)
                background = background.astype(np.uint8)
            
            #img = ((background*np.where(building_mask[:, :, 0]==0, 1, 0)) + building_mask).astype(np.uint8)
            mask_tmp = np.zeros((self.IMG_ROW, self.IMG_COL, self.num_channels))
            mask_tmp[:,:,0] = mask[:, :, 0]==0
            mask_tmp[:,:,1] = mask[:, :, 0]==0
            mask_tmp[:,:,2] = mask[:, :, 0]==0
            # burde være 12 kanaler
            # mappe masken, 1 om bygning, 0 ellers?
            img = ((background*mask_tmp) + building_mask).astype(np.uint8)
            
            # add shadows
            if self.shadows:
                mask_shift = mask[:, :, 0]*1
        
                shift_x = random.randint(0,6) 
                shift_y = 4

                for i_sh in range(1, shift_x):
                    mask_shift[:-i_sh, :-i_sh-4] += mask[i_sh:, i_sh+4: , 0]

                shadow = (mask_shift>0)* (mask[:, :, 0]==0)
                alpha = random.choice([0.4, .3, .45])
                
                for i in range(3):
                    img[:,:,i] = img[:,:,i]*(shadow==0)+(alpha*(shadow>0)*img[:,:,i])
                
            flag_instance_augm = True
            
        #---------------------------------------------------------------------------------------
        # base augmentation
        #---------------------------------------------------------------------------------------
        #if self.val:
        #    img, mask_tmp  = augmentation(img, mask, 1.)
        #    if len(mask_tmp.shape)==2:
        #        mask[:,:,0]=mask_tmp
        #    else:
        #        mask=mask_tmp
                
        if self.augm and not flag_instance_augm:
            # if instance has not already been augmented, but should be augmented
            img, mask_tmp  = augment(img, mask, self.color_aug_prob)
            if len(mask_tmp.shape) == 2:
                mask[:,:,0]=mask_tmp
            else:
                mask=mask_tmp
                
        img = img / 255. # normalize?
        img = img.clip(0, 1) # setter verdier mellom 0 og 1
        return np.asarray(img), np.asarray(mask) #, np.asarray(background/ 255.), np.asarray(img_initial/ 255.)
    

    
    def extract_val(self, sample):
        return sample['upper_left_x'], sample['upper_left_y'], sample['pol_width'], sample['pol_height']
    
    def train_gen(self):
        while True:
            self.val = False
            #self.background_prob = 0.5
            imgarr=[]
            maskarr=[]
            for i in range(self.batch_size):
                random_key = random.choice(list(self.json_file_cl0_train.keys()))
                upper_left_x, upper_left_y, pol_width, pol_height = self.extract_val(self.json_file_cl0_train[random_key])
                
                img_name = random_key[:-len(random_key.split('_')[-1])-1]
                img,mask=self.get_img_mask_array(img_name, upper_left_x, upper_left_y, 
                                                 pol_width, pol_height, random_key.split('_')[-1])
                imgarr.append(img)
                maskarr.append(mask)
            yield (np.asarray(imgarr),np.asarray(maskarr))
            imgarr=[]
            maskarr=[] 

    def val_gen(self):
        while True:
            self.val = True
            #self.background_prob = 0.
            imgarr=[]
            maskarr=[]
            #background = []
            #img_initial = []
            for i in range(self.batch_size):
                random_key = random.choice(list(self.json_file_cl0_val.keys()))
                upper_left_x, upper_left_y, pol_width, pol_height = self.extract_val(self.json_file_cl0_val[random_key])
                
                img_name = random_key[:-len(random_key.split('_')[-1])-1]
                img,mask =self.get_img_mask_array(img_name, upper_left_x, upper_left_y, 
                                                 pol_width, pol_height, random_key.split('_')[-1])
                #background.append(background_img)
                maskarr.append(mask)
                imgarr.append(img)
                #img_initial.append(initial)
            yield (np.asarray(imgarr),np.asarray(maskarr)) #, np.asarray(background), np.asarray(img_initial))
            imgarr=[]
            maskarr=[]
   
    def read_json(self, folder):
        js_full = {}
        json_file = '{}/{}.json'.format(folder, "train_annotations")
        with open(json_file, 'r') as f:
            js_tmp = json.load(f)
        keys_list = set(js_tmp.keys())
        for key in keys_list:
            js_tmp[folder+'_'+key] = js_tmp[key]
            del js_tmp[key]
        js_full.update(js_tmp)
        return js_full    

    def train_val_split_prob(self, js_full, split_ration):                
        seed(1)
        train_samples, val_samples = {}, {}
        keys_list = set(js_full.keys())
        for key in keys_list:
            if random.random() < split_ration:
                train_samples[key] = js_full[key]
            else:
                val_samples[key] = js_full[key]
            del js_full[key]

        return train_samples, val_samples
    

    # split dataset: this is to be done in the loader, not needed? 
    def train_val_split(self, js_full, split_ration):               
        seed(1)
        train_samples, val_samples = {}, {}
        keys_list = set(js_full.keys())
        for key in keys_list:
            if js_full[key]["upper_left_y"] > self.train_upper_threshold and js_full[key]["upper_left_y"] < self.train_lower_threshold: # this threshold is for Venture image
                train_samples[key] = js_full[key]
            elif js_full[key]["upper_left_y"] < self.val_lower_threshold and js_full[key]["upper_left_y"] > self.val_upper_threshold:
                val_samples[key] = js_full[key]
            del js_full[key]
        return train_samples, val_samples

    
    def load_dataset(self, folder, folders_val = None, split_ration=0.7):
        self.json_file_train = self.read_json(folder)
        
        if folders_val:
            self.json_file_val = self.read_json(folders_val)

        # elif self.prob_split: # prob_split?
        #     self.json_file_train, self.json_file_val = self.train_val_split_prob(self.read_json(folder), split_ration)
        else:
            self.json_file_train, self.json_file_val = self.train_val_split(self.read_json(folder), split_ration)
  

# # Ser ikke ut til å bli brukt noen sted, kun hastaget kode
def rotate(img, angle): # roterer kun bildet, ikke masken
    (height, width) = img.shape[:2]
    (cent_x, cent_y) = (width // 2, height // 2)

    mat = cv2.getRotationMatrix2D((cent_x, cent_y), -angle, 1.0)
    cos = np.abs(mat[0, 0])
    sin = np.abs(mat[0, 1])

    n_width = int((height * sin) + (width * cos))
    n_height = int((height * cos) + (width * sin))

    mat[0, 2] += (n_width / 2) - cent_x
    mat[1, 2] += (n_height / 2) - cent_y

    return cv2.warpAffine(img, mat, (n_width, n_height))