import argparse
import numpy as np
import cv2
import os
from config import *
from model import ZSSR
from data_utils import *
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr
import glob



# main
if __name__ == '__main__':
    np.random.seed(0)
    if keras.backend == 'tensorflow':
        keras.backend.set_image_dim_ordering('tf')
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # Provide an alternative to provide MissingLinkAI credential
    parser = argparse.ArgumentParser()
    parser.add_argument('--srFactor', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--filepath', type=str, default='imgs/img_001_SRF_2_LR.png')
    parser.add_argument('--filters', type=int, default=64)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--sortOrder', default='A')
    parser.add_argument('--scalingSteps', type=int, default=1)
    parser.add_argument('--groundTruth', type=bool, default=True)
    parser.add_argument('--flip', type=bool, default=True)
    parser.add_argument('--noiseFlag', type=bool, default=True)
    parser.add_argument('--noiseSTD', type = int, default=30)
    parser.add_argument('--save_aug', type = bool, default=True)
    parser.add_argument('--output_paths', type=str, default='output')
    # Override credential values if provided as arguments
    args = parser.parse_args()

    
    file_name = args.filepath
    SR_FACTOR = args.srFactor
    FILTERS = args.filters
    EPOCHS = args.epochs 
    SHUFFLE = args.shuffle
    BATCH_SIZE = args.batch 
    LAYERS_NUM = args.layers 
    SORT_ORDER = args.sortOrder 
    NB_SCALING_STEPS = args.scalingSteps
    GROUND_TRUTH = args.groundTruth 
    BASELINE = args.baseline 
    FLIP_FLAG = args.flip 
    NOISE_FLAG = args.noiseFlag
    NOISY_PIXELS_STD = args.noiseSTD
    SAVE_AUG  = args.save_aug
    # We're making sure These parameters are equal, in case of an update from the parser.


    # Path for Data and Output directories on Docker
    # save to disk
    if not os.path.exists(output_paths):
        os.makedirs(output_paths)

    # Load image from data volumes
    image = load_img(file_name)
    cv2.imwrite(output_paths + '/'  + 'image.png',
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR), params=[CV_IMWRITE_PNG_COMPRESSION])


    # Build and compile model
    X = Input(shape=(None, None , 3))
    zssr = ZSSR().zssr(X)
    # Learning rate scheduler
    lrate = LearningRateScheduler(step_decay)

    callbacksList = [lrate] #, checkpoint
    # TRAIN
    history = zssr.fit_generator(image_generator(image, EPOCHS, BATCH_SIZE, NB_SCALING_STEPS),
                                 steps_per_epoch=NB_STEPS, 
                                 epochs=EPOCHS, 
                                 shuffle=SHUFFLE, 
                                 callbacks=callbacksList,
                                 max_queue_size=32,
                                 verbose=1)
    # Saving our model and weights
    zssr.save(output_paths + '/zssr_model.h5')
    # PREDICT
    super_image, interpolated_image = ZSSR().predict_func(zssr, image)
    # Get super resolution images
    super_image_accumulated_median, super_image_accumulated_avg = ZSSR().accumulated_result(zssr, image)