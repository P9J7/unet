from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans
import keras.preprocessing.image as kpi

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)


def overlapData(img,target_size=(572, 572)):
    img = np.reshape(img, (img.shape[1], img.shape[2]))
    # print(img.shape[0])
    # print(target_size[0])
    tailSize = (target_size[0] - img.shape[0]) // 2
    # print(tailSize)
    tailImg = np.zeros(target_size, dtype=float)
    tailImg[tailSize:img.shape[0] + tailSize, tailSize:img.shape[0] + tailSize] = img
    # left
    reversedCol = np.arange(tailSize - 1, -1, -1)
    tailImg[tailSize:img.shape[0] + tailSize, :tailSize] = img[:, reversedCol]
    # right
    reversedCol = np.arange(img.shape[1] - 1, img.shape[1] - 1 - tailSize, -1)
    tailImg[tailSize:img.shape[0] + tailSize, img.shape[0] + tailSize:] = img[:, reversedCol]
    # up
    reversedRow = np.arange(tailSize * 2 - 1, tailSize - 1, -1)
    tailImg[0:tailSize, :] = tailImg[reversedRow, :]
    # down
    reversedRow = np.arange(img.shape[0] + tailSize - 1, img.shape[0] - 1, -1)
    tailImg[img.shape[1] + tailSize:, :] = tailImg[reversedRow, :]
    tailImg = np.reshape(tailImg, (1,) + tailImg.shape + (1,))
    return tailImg


def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)


def overlapTrainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (512,512),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        img,mask = overlapData(img),mask
        yield img,mask
        # saveTrainResult("data/membrane/train/overlapTrain", img, mask)


def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        # saveResult("data/membrane/train/unetTest", img)
        yield img


def overlapTestGenerator(test_path,num_image=30, target_size=(572, 572), flag_multi_class=False, as_gray=True):
    for i in range(num_image):
        # img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        img_path = os.path.join(test_path,"%d.png"%i)
        img = kpi.load_img(img_path, color_mode="grayscale", target_size=(388, 388))
        img = kpi.img_to_array(img, 'channels_last')
        img = img[:, :, 0]
        img = img / 255
        tailSize = (target_size[0] - img.shape[0]) // 2
        # print(tailSize)
        tailImg = np.zeros(target_size, dtype=float)
        tailImg[tailSize:img.shape[0] + tailSize, tailSize:img.shape[0] + tailSize] = img
        # left
        reversedCol = np.arange(tailSize - 1, -1, -1)
        tailImg[tailSize:img.shape[0] + tailSize, :tailSize] = img[:, reversedCol]
        # right
        reversedCol = np.arange(img.shape[1] - 1, img.shape[1] - 1 - tailSize, -1)
        tailImg[tailSize:img.shape[0] + tailSize, img.shape[0] + tailSize:] = img[:, reversedCol]
        # up
        reversedRow = np.arange(tailSize*2 - 1, tailSize - 1, -1)
        tailImg[0:tailSize, :] = tailImg[reversedRow, :]
        # down
        reversedRow = np.arange(img.shape[0] + tailSize - 1, img.shape[0] - 1, -1)
        tailImg[img.shape[1] + tailSize:, :] = tailImg[reversedRow, :]
        tailImg = np.reshape(tailImg, tailImg.shape+(1,)) if (not flag_multi_class) else tailImg
        tailImg = np.reshape(tailImg, (1,)+tailImg.shape)
        # saveTrainResult("data/membrane/train/overlapTest", tailImg)
        yield tailImg


def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255



def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        # print(item[:, :, 0])
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)


def saveTrainResult(save_path,img,mask=None):
    img = img[0,:, :, 0]
    i = np.random.randint(10000)
    io.imsave(os.path.join(save_path, "%d_train.png" % i), img)
    if (mask != None):
        mask = mask[0, :, :, 0]
        io.imsave(os.path.join(save_path, "%d_mask.png" % i), mask)
