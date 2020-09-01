import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input
from config import *


class ZSSR(object):   
    def zssr(self, X):
        # model
        inter = Conv2D(filters=FILTERS, kernel_size=3, activation='relu', padding = 'same', strides=1)(X)

        # Create inner Conv Layers
        for layer in range(LAYERS_NUM):
            inter = Conv2D(filters=FILTERS, kernel_size=3, strides=1, padding='same', activation='relu')(inter)

        inter = Conv2D(filters=3, kernel_size=3, strides=1, padding='same', activation="linear")(inter)

        # Residual layer
        out = inter + X

        zssr_model = Model(inputs=X, outputs=out)

        zssr_model.compile(loss='mae', optimizer='adam')

        zssr_model.summary()
        return zssr_model

    def predict_func(self, model, image):
        # Resize original image to super res size
        interpolated_image = cv2.resize(image, None, fx=SR_FACTOR, fy=SR_FACTOR, interpolation=cv2.INTER_CUBIC)  # SR_FACTOR
        # Expand dims for NN
        interpolated_image = np.expand_dims(interpolated_image, axis=0)

        # Expand dims for NN
        # Check if image is a 4D tensor
        if len(np.shape(interpolated_image)) == 4:
            pass
        else:
            interpolated_image = np.expand_dims(interpolated_image, axis=0)
        # Get prediction from NN
        super_image = model.predict(interpolated_image)
        # Reduce the unwanted dimension and get image from tensor
        super_image = np.squeeze(super_image, axis=(0))
        interpolated_image = np.squeeze(interpolated_image, axis=(0))

        # Normalize data type back to uint8
        super_image = cv2.convertScaleAbs(super_image)
        interpolated_image = cv2.convertScaleAbs(interpolated_image)

        # Save super res image
        cv2.imwrite(output_paths + '/' + str(SR_FACTOR) + '_super.png', cv2.cvtColor(super_image, cv2.COLOR_RGB2BGR),
                    params=[CV_IMWRITE_PNG_COMPRESSION])
        # Save bi-cubic enlarged image
        cv2.imwrite(output_paths + '/' + str(SR_FACTOR) + '_super_size_interpolated.png',
                    cv2.cvtColor(interpolated_image, cv2.COLOR_RGB2BGR), params=[CV_IMWRITE_PNG_COMPRESSION])

        return super_image, interpolated_image


    def accumulated_result(self, model, image):
        # Resize original image to super res size
        int_image = cv2.resize(image, None, fx=SR_FACTOR, fy=SR_FACTOR, interpolation=cv2.INTER_CUBIC)  # SR_FACTOR
        print("NN Input shape:", np.shape(int_image))
        super_image_accumulated = np.zeros(np.shape(int_image))
        # Get super res image from the NN's output
        super_image_list = []
        for k in range(0, 8):
            print("k", k)
            img = np.rot90(int_image, k, axes=(0, 1))
            if (k > 3):
                print("flip")
                img = np.fliplr(img)
            # Expand dims for NN
            img = np.expand_dims(img, axis=0)
            super_img = model.predict(img)
            super_img = np.squeeze(super_img, axis=(0))
            super_img = cv2.convertScaleAbs(super_img)
            if (k > 3):
                print("unflip")
                super_img = np.fliplr(super_img)
            super_img = np.rot90(super_img, -k, axes=(0, 1))
            super_image_list.append(super_img)
            super_image_accumulated = super_image_accumulated + super_img

        super_image_accumulated_avg = np.divide(super_image_accumulated, 8)
        # Normalize data type back to uint8
        super_image_accumulated_avg = cv2.convertScaleAbs(super_image_accumulated_avg)
        cv2.imwrite(output_paths + '/' + str(SR_FACTOR) + '_super_image_accumulated_avg.png',
                    cv2.cvtColor(super_image_accumulated_avg, cv2.COLOR_RGB2BGR), params=[CV_IMWRITE_PNG_COMPRESSION])

        super_image_accumulated_median = np.median(super_image_list, axis=0)
        super_image_accumulated_median = cv2.convertScaleAbs(super_image_accumulated_median)
        cv2.imwrite(output_paths + '/' + str(SR_FACTOR) + '_super_image_accumulated_median.png',
                    cv2.cvtColor(super_image_accumulated_median, cv2.COLOR_RGB2BGR), params=[CV_IMWRITE_PNG_COMPRESSION])

        return super_image_accumulated_median, super_image_accumulated_avg