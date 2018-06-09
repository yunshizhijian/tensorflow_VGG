import inspect
import os
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

VGG_MEAN = [103.939, 116.779, 123.68] #样本RGB的平均值

class Vgg16():
    def __init__(self, vgg16_path=None):
        if vgg16_path is None:
            vgg16_path = os.path.join(os.getcwd(), "vgg16.npy")  #返回当前的工作目录
            print(vgg16_path)
            #遍历键值对,导入模型参数
            self.data_dict = np.load(vgg16_path, encoding='latin1').item()
        for x in self.data_dict: #遍历data_dict中的每个键
            print(x)

    def forward(self, images):
        print("build model started")
        start_time = time.time() #获取前向传播的开始时间
        rgb_scaled = images * 255.0 #按照逐个像素乘以255.从GRB变为BGR
        red, green, blue = tf.split(rgb_scaled,3,3) 
        bgr = tf.concat([     
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2]],3)
        #接下来构造VGG的16层网络(5层卷积,3层全连接),命名规范
        #1:第一层卷积,含两个卷积层,后边跟随池化层,缩小图片尺寸
        self.conv1_1 = self.conv_layer(bgr, "conv1_1") 
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool_2x2(self.conv1_2, "pool1")

        #传入第一层的迟化结果,获取该层的卷积和偏置,再去卷积运算,最后返回激活函数后的值
        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool_2x2(self.conv2_2, "pool2")

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool_2x2(self.conv3_3, "pool3")
        
        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool_2x2(self.conv4_3, "pool4")
        
        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool_2x2(self.conv5_3, "pool5")

        #根据上一层的池化层输出来进行加权和求运算
        self.fc6 = self.fc_layer(self.pool5, "fc6")
        self.relu6 = tf.nn.relu(self.fc6) 
        
        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        #最后一层全连接后,实现softmax分类,得到各分类的概率
        self.fc8 = self.fc_layer(self.relu7, "fc8")
        self.prob = tf.nn.softmax(self.fc8, name="prob")

        #结束前向传播的时间
        end_time = time.time() 
        print(("time consuming: %f" % (end_time-start_time)))
        #清空本次独到的模型参数字典
        self.data_dict = None 
        
    def conv_layer(self, x, name):
        with tf.variable_scope(name): 
            w = self.get_conv_filter(name) #读到该层的卷积核
            conv = tf.nn.conv2d(x, w, [1, 1, 1, 1], padding='SAME') #进行卷积计算
            conv_biases = self.get_bias(name) #读到偏置项
            result = tf.nn.relu(tf.nn.bias_add(conv, conv_biases)) #加上偏置后进行激活
            return result
    
    def get_conv_filter(self, name):
        #获得VGG16.npy中的卷积核
        return tf.constant(self.data_dict[name][0], name="filter") 
    
    def get_bias(self, name):
        #获得该卷积的偏置项
        return tf.constant(self.data_dict[name][1], name="biases")
    
    def max_pool_2x2(self, x, name):
        #定义最大层的池化层操作
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    #定义全连接层的前向传播
    def fc_layer(self, x, name):
        with tf.variable_scope(name):#根据命名空间做全连接层计算
            shape = x.get_shape().as_list() #获取该层的维度信息列表
            dim = 1
            for i in shape[1:]:
                dim *= i
            #改变特征图形状,多维特征进行拉伸操作
            x = tf.reshape(x, [-1, dim])
            w = self.get_fc_weight(name) 
            b = self.get_bias(name) 
                
            result = tf.nn.bias_add(tf.matmul(x, w), b) 
            return result
    
    def get_fc_weight(self, name):  
        return tf.constant(self.data_dict[name][0], name="weights")

