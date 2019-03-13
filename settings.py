# content path
CONTENT_IMAGE = 'images/content.jpg'
# style path
STYLE_IMAGE = 'images/style.jpg'
# output path
OUTPUT_IMAGE = 'output/output'
# pre_trained_vgg19
VGG_MODEL_PATH = 'imagenet-vgg-verydeep-19.mat'

IMAGE_WIDTH = 450

IMAGE_HEIGHT = 300

CONTENT_LOSS_LAYERS = [('relu4_2', 0.5),('relu5_2',0.5)]
STYLE_LOSS_LAYERS = [('relu1_1', 0.2), ('relu2_1', 0.2), ('relu3_1', 0.2), ('relu4_1', 0.2), ('relu5_1', 0.2)]
NOISE = 0.5
#
IMAGE_MEAN_VALUE = [128.0, 128.0, 128.0]
# weights for content
ALPHA = 1
# weights for style
BETA = 500
