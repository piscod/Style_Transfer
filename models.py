import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc
import settings


class Model(object):
    def __init__(self, content_path, style_path):
        self.content = self.loadimg(content_path)  # load content img
        self.style = self.loadimg(style_path)  # load style img
        self.random_img = self.get_random_img()  # get noise img
        self.model = self.vggnet()  # bulit up vgg19 network

    def vggnet(self):
        
        vgg_model = scipy.io.loadmat(settings.VGG_MODEL_PATH)  # load vgg parameters
        vgg_layers = vgg_model['layers']

        model = {}
        """Here is the detailed configuration of the VGG model:
        0 is conv1_1 (3, 3, 3, 64)
        1 is relu
        2 is conv1_2 (3, 3, 64, 64)
        3 is relu    
        4 is maxpool
        5 is conv2_1 (3, 3, 64, 128)
        6 is relu
        7 is conv2_2 (3, 3, 128, 128)
        8 is relu
        9 is maxpool
        10 is conv3_1 (3, 3, 128, 256)
        11 is relu
        12 is conv3_2 (3, 3, 256, 256)
        13 is relu
        14 is conv3_3 (3, 3, 256, 256)
        15 is relu
        16 is conv3_4 (3, 3, 256, 256)
        17 is relu
        18 is maxpool
        19 is conv4_1 (3, 3, 256, 512)
        20 is relu
        21 is conv4_2 (3, 3, 512, 512)
        22 is relu
        23 is conv4_3 (3, 3, 512, 512)
        24 is relu
        25 is conv4_4 (3, 3, 512, 512)
        26 is relu
        27 is maxpool
        28 is conv5_1 (3, 3, 512, 512)
        29 is relu
        30 is conv5_2 (3, 3, 512, 512)
        31 is relu
        32 is conv5_3 (3, 3, 512, 512)
        33 is relu
        34 is conv5_4 (3, 3, 512, 512)
        35 is relu
        36 is maxpool
        37 is fullyconnected (7, 7, 512, 4096)
        38 is relu
        39 is fullyconnected (1, 1, 4096, 4096)
        40 is relu
        41 is fullyconnected (1, 1, 4096, 1000)
        42 is softmax"""

        model['input'] = tf.Variable(np.zeros((1, settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH, 3)), dtype='float32')

        model['relu1_1'] = tf.nn.relu(self.conv2d(vgg_layers,model['input'], 0))
        model['relu1_2'] = tf.nn.relu(self.conv2d(vgg_layers,model['relu1_1'], 2))
        model['pool1'] = tf.nn.avg_pool(model['relu1_2'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        model['relu2_1'] = tf.nn.relu(self.conv2d(vgg_layers,model['pool1'], 5))
        model['relu2_2'] = tf.nn.relu(self.conv2d(vgg_layers,model['relu2_1'], 7))
        model['pool2'] = tf.nn.avg_pool(model['relu2_2'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        model['relu3_1'] = tf.nn.relu(self.conv2d(vgg_layers,model['pool2'], 10))
        model['relu3_2'] = tf.nn.relu(self.conv2d(vgg_layers,model['relu3_1'], 12))
        model['relu3_3'] = tf.nn.relu(self.conv2d(vgg_layers,model['relu3_2'], 14))
        model['relu3_4'] = tf.nn.relu(self.conv2d(vgg_layers,model['relu3_3'], 16))
        model['pool3'] = tf.nn.avg_pool(model['relu3_4'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        model['relu4_1'] = tf.nn.relu(self.conv2d(vgg_layers,model['pool3'], 19))
        model['relu4_2'] = tf.nn.relu(self.conv2d(vgg_layers,model['relu4_1'], 21))
        model['relu4_3'] = tf.nn.relu(self.conv2d(vgg_layers,model['relu4_2'], 23))
        model['relu4_4'] = tf.nn.relu(self.conv2d(vgg_layers,model['relu4_3'], 25))
        model['pool4'] = tf.nn.avg_pool(model['relu4_4'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        model['relu5_1'] = tf.nn.relu(self.conv2d(vgg_layers,model['pool4'], 28))
        model['relu5_2'] = tf.nn.relu(self.conv2d(vgg_layers,model['relu5_1'], 30))
        model['relu5_3'] = tf.nn.relu(self.conv2d(vgg_layers,model['relu5_2'], 32))
        model['relu5_4'] = tf.nn.relu(self.conv2d(vgg_layers,model['relu5_3'], 34))
        model['pool5'] = tf.nn.avg_pool(model['relu5_4'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # end up with here without Fully connected layers

        return model
            
        
    def conv2d(self, vgg_layer,previous_layer, layer):
        '''get the results of convolution layer'''
        vgg_layers = vgg_layer
        W = vgg_layers[0][layer][0][0][2][0][0]  # details in explore_pretrained_vgg
        b = vgg_layers[0][layer][0][0][2][0][1]
        layer_name = vgg_layers[0][layer][0][0][0][0]
        conv = tf.nn.conv2d(previous_layer, filter=tf.constant(W), strides=[1, 1, 1, 1], padding='SAME')
        bias = tf.constant(np.reshape(b, b.size))

        return conv + bias


    def loadimg(self, path):
        # load img
        image = scipy.misc.imread(path)
        # resize img
        image = scipy.misc.imresize(image, [settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH])
        # batch=1
        image = np.reshape(image, (1, settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH, 3))
        # substract mean
        image = image - settings.IMAGE_MEAN_VALUE

        return image

    def get_random_img(self):
        # same shape as loading img
        noise_image = np.random.uniform(-20, 20, [1, settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH, 3])
        random_img = noise_image * settings.NOISE + self.content * (1 - settings.NOISE)
        return random_img

if __name__ == '__main__':
    Model(settings.CONTENT_IMAGE, settings.STYLE_IMAGE)
