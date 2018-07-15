import keras
import matplotlib.pyplot as plt
import urllib.request
import numpy as np
import cv2
import glob
from net_def import build_model
from io import StringIO, BytesIO
from PIL import Image
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import *
from keras.optimizers import Adam



model = build_model('cnt-k2-WorldExpo-AFCN-7c-0003-LRN_v2-my-mask-step_lr-larger_loss_weight.yaml', 'model_weights_iter_0600000.h5')

#Crop image to pieces
# dir_path = './static/imageset/'
# img_path= './static/images/test2.png'
# # img_path = BytesIO(urllib.request.urlopen(img_path).read())
# def crop(path, input, height, width, k, page, area):
#     im = Image.open(input)
#     imgwidth, imgheight = im.size
#     for i in range(0,imgheight,height):
#         for j in range(0,imgwidth,width):
#             box = (j, i, j+width, i+height)
#             a = im.crop(box)
#             try:
#                 o = a.crop(area)
#                 plt.imshow(o)
#                 plt.savefig(path + ("IMG-%s.png" % k))
#             except:
#                 pass
#             k +=1
#
# crop(dir_path, img_path, 128, 128, 135, 'test', (0, 0, 128, 128))

# path = './static/train/'
# gen_path = './static/result/'
#
# def print_result(path):
#     name_list = glob.glob(path)
#     fig = plt.figure()
#     for i in range(9):
#         img = Image.open(name_list[i])
#         # add_subplot(331) 参数一：子图总行数，参数二：子图总列数，参数三：子图位置
#         sub_img = fig.add_subplot(331 + i)
#         sub_img.imshow(img)
#     plt.show()
#     return fig
#
#
# name_list = glob.glob(path + '*/*')
# # print(name_list)
#
#
# fig = print_result(path + '*/*')
#
#
# fig.savefig(gen_path + '/original_0.png', dpi=200, papertype='a5')
#
# datagen = image.ImageDataGenerator()
# gen_data = datagen.flow_from_directory(path,
#                             target_size=(128, 128),
#                             batch_size=32, shuffle=False,
#                             save_to_dir=gen_path,
#                             save_prefix='train_gan')
#
# for i in range(9):
#     gen_data.next()
#
# fig = print_result(gen_path + '/*')
# fig.savefig(gen_path + '/original_1.png', dpi=200, papertype='a5')

datagen = image.ImageDataGenerator()

train_path = './static/train'
valid_path = './static/valid'
test_path ='./static/test'

train_batches = datagen.flow_from_directory(train_path, target_size=(128,128), classes=['human', 'other'], batch_size=32)
train_batches_32 = datagen.flow_from_directory(train_path, target_size=(32,32), classes=['human', 'other'], batch_size=32)
valid_batches = datagen.flow_from_directory(valid_path, target_size=(128,128), classes=['human', 'other'], batch_size=32)
test_batches = datagen.flow_from_directory(test_path, target_size=(128,128), classes=['human', 'other'], batch_size=32)

aux = np.full((32), -60)

output = np.vstack([train_batches, aux, train_batches_32])

model.summary()

model.fit_generator(train_batches, steps_per_epoch=2, validation_data=valid_batches, validation_steps=2, epochs=5)
