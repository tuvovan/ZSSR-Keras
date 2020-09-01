import argparse
import numpy as np
import cv2
import os
import glob
from config import *

from tensorflow import keras
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr


""" Load an image."""
def load_img(file_name):
    # Load the image
    image = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = np.stack((image,) * 3, axis=-1)

    image = image.astype('float32')
    return image

# Add noise to lr sons
def add_noise(image):
    row, col, ch = image.shape

    noise = np.random.normal(0, NOISY_PIXELS_STD, (row, col, ch))
    #Check image dtype before adding.
    noise = noise.astype('float32')
    # We clip negative values and set them to zero and set values over 255 to it.
    noisy = np.clip((image + noise), 0, 255)

    return noisy


def preprocess(image, scale_fact, scale_fact_inter, i):

    # scale down is sthe inverse of the intermediate scaling factor
    scale_down = 1 / scale_fact_inter
    # Create hr father by downscaling from the original image
    hr = cv2.resize(image, None, fx=scale_fact, fy=scale_fact, interpolation=cv2.INTER_CUBIC)

    if CROP_FLAG:
        h_crop = w_crop = np.random.choice(CROP_SIZE)
        #print("h_crop, w_crop:", h_crop, w_crop)
        if (hr.shape[0] > h_crop):
            x0 = np.random.randint(0, hr.shape[0] - h_crop)
            h = h_crop
        else:
            x0 = 0
            h = hr.shape[0]
        if (hr.shape[1] > w_crop):
            x1 = np.random.randint(0, hr.shape[1] - w_crop)
            w = w_crop
        else:
            x1 = 0
            w = hr.shape[1]
        hr = hr[x0 : x0 + h, x1 : x1 + w]

    if FLIP_FLAG:
        # flip
        k = np.random.choice(8)
        hr = np.rot90(hr, k, axes=(0, 1))
        if (k > 3):
            hr = np.fliplr(hr)

    # hr is cropped and flipped then copies as lr
    lr = cv2.resize(hr, None, fx=scale_down, fy=scale_down, interpolation=cv2.INTER_CUBIC)
    # Upsample lr to the same size as hr
    lr = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Add gaussian noise to the downsampled lr
    if NOISE_FLAG:
        lr = add_noise(lr)
    # Save every Nth augmentation to artifacts.
    if SAVE_AUG and i%50==0:
        dir_name = os.path.join(output_paths, 'Aug/')
        # Create target Directory if don't exist
        if os.path.isdir(dir_name):
            pass
        else:
            os.mkdir(dir_name)

        cv2.imwrite(output_paths + '/Aug/' + str(SR_FACTOR) + '_' + str(i) + 'lr.png', cv2.cvtColor(lr, cv2.COLOR_RGB2BGR), params=[CV_IMWRITE_PNG_COMPRESSION])
        cv2.imwrite(output_paths + '/Aug/' + str(SR_FACTOR) + '_' + str(i) + 'hr.png', cv2.cvtColor(hr, cv2.COLOR_RGB2BGR), params=[CV_IMWRITE_PNG_COMPRESSION])

    # Expand image dimension to 4D Tensors.
    lr = np.expand_dims(lr, axis=0)
    hr = np.expand_dims(hr, axis=0)

    X = lr
    y = hr

    return X, y


def s_fact(image, NB_PAIRS, NB_SCALING_STEPS):
    BLUR_LOW_BIAS = 0.0
    scale_factors = np.empty(0)
    if image.shape[0] * image.shape[1] <= 50 * 50:
        BLUR_LOW_BIAS = 0.3
    for i in range(NB_SCALING_STEPS):
        temp = np.random.uniform(BLUR_LOW + BLUR_LOW_BIAS, BLUR_HIGH,
                                 int(NB_PAIRS / NB_SCALING_STEPS))  # Low = 0.4, High = 0.95
        if SORT:
            temp = np.sort(temp)
        if SORT_ORDER == 'D':
            temp = temp[::-1]
        scale_factors = np.append(scale_factors, temp, axis=0)
        scale_factors = np.around(scale_factors, decimals=5)
    scale_factors_pad = np.repeat(scale_factors[-1], abs(NB_PAIRS - len(scale_factors)))
    scale_factors = np.concatenate((scale_factors, scale_factors_pad), axis=0)

    # Intermediate SR_Factors
    intermidiate_SR_Factors = np.delete(np.linspace(1, SR_FACTOR, NB_SCALING_STEPS + 1), 0)
    intermidiate_SR_Factors = np.around(intermidiate_SR_Factors, decimals=3)

    lenpad = np.int(NB_PAIRS / NB_SCALING_STEPS)
    intermidiate_SR_Factors = np.repeat(intermidiate_SR_Factors, lenpad)

    pad = np.repeat(intermidiate_SR_Factors[-1], abs(len(intermidiate_SR_Factors) - max(len(scale_factors), NB_PAIRS)))
    intermidiate_SR_Factors = np.concatenate((intermidiate_SR_Factors, pad), axis=0)
    return scale_factors, intermidiate_SR_Factors


def image_generator(image, EPOCHS, BATCH_SIZE, NB_SCALING_STEPS):
    i = 0
    scale_fact, scale_fact_inter = s_fact(image, NB_PAIRS, NB_SCALING_STEPS)
    while i < EPOCHS:
        X, y = preprocess(image, scale_fact[i] + np.round(np.random.normal(0.0, 0.03), decimals=3),
                          scale_fact_inter[i], i)

        i = i + 1

        yield X, y

def step_decay(epochs):
    initial_lrate = INITIAL_LRATE
    drop = DROP
    if LEARNING_RATE_CYCLES:
        cycle = np.ceil(NB_PAIRS / NB_SCALING_STEPS)
        epochs_drop = np.ceil((NB_STEPS * EPOCHS) / NB_SCALING_STEPS)
        step_length = int(epochs_drop / FIVE)
    else:
        cycle = NB_PAIRS
        epochs_drop = np.ceil((NB_STEPS * EPOCHS) / FIVE)
        step_length = epochs_drop

    lrate = initial_lrate * np.power(drop, np.floor((1 + np.mod(epochs, cycle)) / step_length))
    return lrate

def ssim_calc(img1, img2, scaling_fact):
    ssim_sk = ssim(img1, img2,
                   data_range=img1.max() - img1.min(), multichannel=True)
    print("SSIM:", ssim_sk)

    return ssim_sk


def psnr_calc(img1, img2, scaling_fact):
    # Get psnr measure from skimage lib
    PIXEL_MAX = 255.0
    PIXEL_MIN = 0.0
    sk_psnr = compare_psnr(img1, img2, data_range=PIXEL_MAX - PIXEL_MIN)
    print("PSNR:", sk_psnr)

    return sk_psnr


def metric_results(ground_truth_image, super_image):
    try:
        ground_truth_image
        psnr_score = psnr_calc(ground_truth_image, super_image, SR_FACTOR)
        ssim_score = ssim_calc(ground_truth_image, super_image, SR_FACTOR)

    except NameError:
        psnr_score = None
        ssim_score = None

    return psnr_score, ssim_score