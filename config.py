import os
# scaling factor
SR_FACTOR = 2
# Data generator random ordered
SHUFFLE = False
# scaling factors array order random or sorted
SORT = True
# Ascending or Descending: 'A' or 'D'
SORT_ORDER = 'A'
# number of time steps (pairs) per epoch
NB_STEPS = 1
# Batch size
BATCH_SIZE = 1
# Number of channels in signal
NB_CHANNELS = 3
# No. of NN filters per layer
FILTERS = 64  # 64 on the paper
# Number of internal convolutional layers
LAYERS_NUM = 6
# No. of scaling steps. 6 is best value from paper.
NB_SCALING_STEPS = 1
# No. of LR_HR pairs
EPOCHS = NB_PAIRS = 1500
# Default crop size (in the paper: 128*128*3)
CROP_SIZE = [96]#[32,64,96,128]
# Adaptive learning rate
INITIAL_LRATE = 0.001
DROP = 0.5
# Adaptive lrate, Number of learning rate steps (as in paper)
FIVE = 5
# Decide if learning rate should drop in cyclic periods.
LEARNING_RATE_CYCLES = False
#
PLOT_FLAG = False
# Crop image for training
CROP_FLAG = True
# Flip flag
FLIP_FLAG = True
# initial scaling bias (org to fathers)
SCALING_BIAS = 1
# Scaling factors - blurring parameters
BLUR_LOW = 0.4
BLUR_HIGH = 0.95
# Add noise or not to transformations
NOISE_FLAG = False
# Mean pixel noise added to lr sons
NOISY_PIXELS_STD = 30
# Save augmentations
SAVE_AUG = True
# If there's a ground truth image. Add to parse.
GROUND_TRUTH = True
# If there's a baseline image. Add to parse.
BASELINE = True
# png compression ratio: best quality
CV_IMWRITE_PNG_COMPRESSION = 9
#file path
file_name = 'imgs/img_001_SRF_2_LR.png'
output_paths = os.path.join(os.getcwd(), 'output')
ORIGIN_IMAGE = 0
GROUND_TRUTH_IMAGE = 1
BASELINE_IMAGE = 2