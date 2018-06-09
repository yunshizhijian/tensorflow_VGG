from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pylab import mpl

mpl.rcParams['font.sans-serif']=['SimHei'] # 正常显示中文标签
mpl.rcParams['axes.unicode_minus']=False # 正常显示正负号

def load_image(path):
    fig = plt.figure("Centre and Resize")
    img = io.imread(path) #读取图片的路径
    img = img / 255.0     #将图片归一化
    
    ax0 = fig.add_subplot(131)  
    ax0.set_xlabel(u'Original Picture')#添加子标签
    ax0.imshow(img) 

    #图像处理部分
    #把图片的宽和高减去最短的边,并且求均值,取出切分出的中心图像
    short_edge = min(img.shape[:2])
    #Python3中需要加入//来表示相除

    y = (img.shape[0] - short_edge) // 2
    x = (img.shape[1] - short_edge) //2
    crop_img = img[y:y+short_edge, x:x+short_edge] 
    
    ax1 = fig.add_subplot(132) 
    ax1.set_xlabel(u"Centre Picture") 
    ax1.imshow(crop_img)
    #中间图像切分成224*224
    re_img = transform.resize(crop_img, (224, 224))
    
    ax2 = fig.add_subplot(133) 
    ax2.set_xlabel(u"Resize Picture") 
    ax2.imshow(re_img)

	#切分好的图片
    img_ready = re_img.reshape((1, 224, 224, 3))
    return img_ready

def percent(value):
    return '%.2f%%' % (value * 100)

