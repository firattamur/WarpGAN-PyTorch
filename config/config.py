"""

Configuration hyperparameters for train.

"""


####### INPUT OUTPUT #######

# The name of the current model for output
name = "WarpGAN - Default"

# The interval between writing summary
summary = 100

# The log interval
log     = 10

# Root of WebCaricature dataset
root = "./datasets/"

# Prefix to the image files
prefix = "WebCaricatureAligned256/"

# Training data list
train_txt = "index_txt/train.txt"

# Test data list
test_txt  = "index_txt/test.txt"

# Target image size (h,w) for the input of network
in_width  = 256
in_height = 256

# 3 channels means RGB, 1 channel for grayscale
in_channels  = 3
n_classes    = 126

# Preprocess for training
preprocess_train = [
    ["random_flip"],
    ["standardize", "mean_scale"],
]

# Preprocess for testing
preprocess_test = [
    ["standardize", "mean_scale"],
]

# Number of GPUs
num_gpus = 1

# Number of workers for dataloader
workers = 2

####### NETWORK #######

# The network architecture
network = "models/default.py"

# Dimensionality of the bottleneck layer in discriminator
bottleneck_size = 512

# Dimensionality of the style space
style_size = 8

# train or evaluation
is_train = True

# initial 
initial = 64

# k
k = 64

# number of landmark points
n_ldmark = 16

# checkpoint path
checkpoint_path = "./checkpoints"

####### TRAINING STRATEGY #######

# Optimizer
optimizer = ("ADAM", {"beta1": 0.5, "beta2": 0.9})

# Number of samples per batch
in_batch = 2

# Number of batches per epoch
epoch_size = 5000

# Number of epochs
num_epochs = 50

# learning rate strategy
learning_rate_strategy = "linear"

# learning rate schedule
lr = 0.0001
learning_rate_schedule = {
    "initial": 1 * lr,
    "start": 50000,
    "end_step": 100000,
}

# Restore model
restore_model = "pretrained/discriminator_casia_256/"

# Weight decay for model variables
weight_decay = 1e-4

# Keep probability for dropouts
keep_prob = 1.0

####### LOSS FUNCTION #######

# Weight of the global adversarial loss
coef_adv = 1.0

# Weight of the patch adversarial loss
coef_patch_adv = 2.0

# Weight of the identity mapping loss
coef_idt = 10.0
