# [Mnist数据集识别报告](https://github.com/fzh0627/Mnist)

## 1. [Mnist数据集](http://yann.lecun.com/exdb/mnist/)

### 1.1 数据集介绍 

由[Yann LeCun](http://yann.lecun.com/), Courant Institute, NYU提出。
The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.
> 训练样本集60000，测试样本集10000
> 每一张图片大小为[28, 28]，灰度值范围为[0, 255]
> 样本标签范围为0-9的是个整数（scalar）集合

### 1.2 [TensorFlow2.0下Mnist数据导入](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/datasets/mnist/load_data)
本次大作业是在深度学习框架TensorFlow2.0下设计实现完成的，所以着重以TensorFlow为核心，来详细讲解关于Mnist数据集合的导入操作。
在TensorFlow2.0中关于常用的数据集（比如mnist、cifar10、cifar100、fashion_mnist等）有一个实用API，可以很方便导入在神经网络学习中经常会用到的数据集。
数据导入的代码如下：
```
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
```
在第一次导入此数据集时，需要从Google网站上下载数据集，在下载之后的数据导入中就不需要再次下载，调用以上代码直接就可以导入到程序当中。

### 1.3 [Mnist数据预处理](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/data/Dataset)
TensorFlow中数据是以张量Tensor的形式在程序内部进行传播，所以需要对上述导入到程序中的数据集合进行预处理。预处理主要分为创造数据集（database）、数据集批处理化（batch）、标签数据预处理（onehot）、数据集打乱操作（shuffle）
#### 标签数据预处理
Mnist中的标签数据是一个0-9的整数形式，为了方便计算和理解我们首先将一个整数（scalar）数据转换为一个10维的一个向量，每一个标签对应的位置数据为1，其他位置为0，这样做可以方便理解输出概率以及计算损失函数loss。代码如下：

#### 创造数据集
导入的(x_train, y_train), (x_test, y_test)数据集并不能直接被TensorFlow直接利用，首先需要创造训练和测试数据集：
```
db = tf.data.Dataset.from_tensor_slices((x_train,y_train))
db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
```
#### 数据集批处理化
利用TensorFlow并行计算的特性，我们需要对Tensor批处理化，从而提高训练效率，在此实验中我们选取
```
batchsz = 128
```
#### 数据集打乱操作
在多个Epoch训练周期之后，为了防止神经网络学习到由于Mnist数据集排序的一些特征，我们需要将数据集合进行随机排序打乱操作。这样可以提高训练得到的神经网络的泛化能力。方法如下：

---
利用以上方法生成的数据集是一个iterable可迭代型数据，可以利用python迭代器具体观察其特性，方法如下：
```
next(iter(db))[0].shape 
next(iter(db))[1].shape
```
由上述方法可以得到的数据集的一个batch的输入数据形状为[128, 28, 28]，样本标签为[128, 10],到此数据集创造成功。

## 2. Mnist手写数字数据集识别问题

### 2.1 问题分析与建模
Mnist手写数字识别是一个神经网络学习的一个基本的入门问题，相当于神经网络的“Hello, world!”程序，所以其问题分析也是比较简单。
由Mnist数据集的输入输出格式分别为[128, 28, 28]和[128, 10]，我们可以将这个问题理解为一个十分类的问题，将图片数据输入到一个用于分类的神经网络当中最后通过设定输出一个[128, 10]大小的数据， 根据哪一维度的数据大就判断手写数字图像属于哪一个类别的原则，所以手写数字分类问题的核心之处在于如何设计出一个优秀的神经网络能够快速有效地逼近正确解。
因为该问题被理解为一个分类问题，所以输出层我们可以用softmax函数，将神经网络输出理解为该神经网络理解一个图像类别地概率，loss判别式可以用CrossEntropy损失函数，这样可以互相弥补从而减轻梯度过大或者过小的效应。
关于具体的神经网络设计，我们将在后面章节进行详细介绍。

### 2.2 已有研究进展
关于Mnist数据集的研究伴随神经网络研究的发展历程已经有很多类别的方法了，下面将以表格的形式列出主要Mnist数据集识别的研究历程。
分类器|判别失误率（%）|发表年份|作者
:---:|:---:|:---:|:---:
一层网络结构线性分类器|12|1998|LeCun|
非线性形变KNN算法|0.54|2007|Keyneth|
启发式搜索树算法|0.87|2009|Keyneth|
1000个RBF核线性分类器|3.6|1998|LeCun|
SVM分类器|0.56|2002|DeCoste and Scholkopf|
6层的结构为784-2500-2000-1500-1000-500-10的线性神经网络|0.35|2010|Ciresan|
卷积神经网络LeNet5|0.8|1998|LeCun|
由上述Mnist发展历程可知，当神经网络的深度增加时，神经网络对于Mnist数据集的识别效果明显提高；卷积神经网络的提出推进了神经网络结构发展有了质的提升。

## 3. 研究算法
### 3.1 Dense+ReLU
首先第一种方法时采用的是五层全连接线性Dense层，每一个Dense后面接一个ReLu函数层，保证神经网络能够提取出Mnist的非线性特征。经过五次这样的结构我们得到的数据维度的变换过程为：[batchsz, 25 * 25] -> [batchsz, 256] -> [batchsz, 128] -> [batchsz, 64] -> [batchsz, 32] -> [batchsz, 10]
#### Dense层
每一个Dense层需要经过初始化（init）和调用（call）两个方法，Dense层的两个实现方法如下：
```
# 初始化
self.dense = layers.Dense(units=256)
#调用
x = self.dense(x)
```
#### ReLU层
线性整流函数（Rectified Linear Unit, ReLU）,又称修正线性单元, 是一种人工神经网络中常用的激活函数（activation function），通常指代以斜坡函数及其变种为代表的非线性函数。
比较常用的线性整流函数有斜坡函数 ![cb2584050f414b74aed3456732ec2dd6.svg+xml](en-resource://database/1419:1)，以及带泄露整流函数 (Leaky ReLU)，其中 ![5264e1d4d820fc4756cf77faa96a3281.svg+xml](en-resource://database/1421:1) 为神经元(Neuron)的输入。线性整流被认为有一定的生物学原理，并且由于在实践中通常有着比其他常用激活函数（譬如逻辑函数）更好的效果，而被如今的深度神经网络广泛使用于诸如图像识别等计算机视觉人工智能领域。
ReLu是当代深度学习使用最多的一种神经元激活函数，其意义在于结构简单，并且可以有效提取神经网络的高维非线性特性。下图为ReLU激活层的图像。
![35cfc6492d13acad2afc5be25db35274.svg+xml](en-resource://database/1425:1)
相比于传统的神经网络激活函数，诸如逻辑函数（Logistic sigmoid）和tanh等双曲函数，线性整流函数有着以下几方面的优势：
* 仿生物学原理：相关大脑方面的研究表明生物神经元的信息编码通常是比较分散及稀疏的。通常情况下，大脑中在同一时间大概只有1%-4%的神经元处于活跃状态。使用线性修正以及正则化（regularization）可以对机器神经网络中神经元的活跃度（即输出为正值）进行调试；相比之下，逻辑函数在输入为0时达到 ![cc34358f00cd72df75ef17705a86d0e2.svg+xml](en-resource://database/1427:1)，即已经是半饱和的稳定状态，不够符合实际生物学对模拟神经网络的期望。不过需要指出的是，一般情况下，在一个使用修正线性单元（即线性整流）的神经网络中大概有50%的神经元处于激活态。
* 更加有效率的梯度下降以及反向传播：避免了梯度爆炸和梯度消失问题。
* 简化计算过程：没有了其他复杂激活函数中诸如指数函数的影响；同时活跃度的分散性使得神经网络整体计算成本下降
ReLU在TF2.0中的实现方法为：
```
x = tf.nn.relu(x)
```
### 3.2 LeNet5
![8d6491c1a629debbf16388f19ae8268e.png](en-resource://database/1431:1)
手写字体识别模型LeNet5诞生于1994年，是最早的卷积神经网络之一。LeNet5通过巧妙的设计，利用卷积、参数共享、池化等操作提取特征，避免了大量的计算成本，最后再使用全连接神经网络进行分类识别，这个网络也是最近大量神经网络架构的起点。
LeNet5由7层CNN（不包含输入层）组成，上图中输入的原始图像大小是32×32像素，卷积层用Ci表示，子采样层（pooling，池化）用Si表示，全连接层用Fi表示。下面逐层介绍其作用和示意图上方的数字含义。

#### Conv2D
卷积（convolution）是通过两个函数f和g生成第三个函数的一种数学算子，表征函数f与g经过翻转和平移的重叠部分的面积。
卷积可分为一维卷积和二维卷积，三维卷积，多维卷积操作。二维卷积是我们最常用的也是最重要的，图像的边缘计算和模糊等算法都是基于卷积操作的只不过是对应的不同计算，卷积滤波器不同。
![f8f9e08a44dfea9a0c2877717259ad85.png](en-resource://database/1433:1)
这里的kernel就是卷积核，kernel_size的大小一般是（3， 3）、（5， 5）、（7、7）这里是奇数的原因是因为方便计算。在本次试验中的kernel_size我们分别用到了（5， 5）和（3， 3）。
在TensorFlow中卷积操作也分为初始化和调用两个方法，分别用下面代码实现：
```
self.conv1 = layers.Conv2D(filters=6, kernel_size=5, padding='valid') # 6@24*24
x = self.conv1(x)
```
filters表示卷积层中卷积核的个数，也表示输出层的维度；
padding有'valid'和'same'两个参数，其中'valid'表示从卷积核中心位于图像边缘时开始计算卷积，'same'表示将所有图像的坐标成为卷积操作的中心点进行卷积操作。

#### MaxPool
最大子采样函数取区域内所有神经元的最大值（max-pooling）。以下图为例，输入数据X为4* 4，采样核size为2，stride为2，no padding。输入数据大小类似卷积层的计算方法，（input_width+2* pad-pool_size）/stride+1。前向传播中不仅要计算pool区域内的最大值，还要记录该最大值所在输入数据中的位置，目的是为了在反向传播中，需要把梯度值传到对应最大值所在的位置。
![29c9a089eb2ca3fa4a486d8015f7e8f6.png](en-resource://database/1435:1)
上图为MaxPool的具体实现方法。通过MaxPool方法我们可以提取特征值较大特征点，抛弃一些无用的特征值/
在TF2.0中具体的实现方法为
```
# 初始化
self.maxpool1 = layers.MaxPool2D()
# 调用
x = self.maxpool1(x)
```
如果在初始化时没有设置任何参数的情况下，则kernel_size = (2, 2)，stride = 2，就相当于对feature_map实现了维度缩小为一半的操作，此次试验就是如此。

#### Flatten
由于此次试验中我们最后需要输出的是一个一维的表示概率的向量，所以我们需要将一个二维的数据做铺平化操作。该方法在TensorFlow2.0中的具体实现方法为：
```
# 初始化
self.flatten = layers.Flatten()
# 调用
x = self.flatten(x)
```
#### 高斯全连接GaussianConnection
高斯全连接也就是采用了径向基核函数的网络连接方式，具体计算过程为：
![3069ac4d34a6518435ae0b0761947b80.png](en-resource://database/1437:1)
在TF2.0中的具体实现方法为：
```
class RBF(layers.Layer):
      def __init__(self, input_dim, output_dim):
            super(RBF, self).__init__()
            self.kernel = self.add_variable('w', [input_dim, output_dim])
      def call(self, inputs):
            inputs = tf.expand_dims(inputs, axis=-1)
            out = tf.reduce_sum(tf.pow(inputs-self.kernel, 2), axis=1)return out
```
### 3.3 ResNet
![d6ed82fba85a3c282ff9f362841e479b.png](en-resource://database/1439:1)
深度残差网络ResNet是由华人科学家何凯明在微软亚洲研究院工作期间提出的。
深度网络容易造成梯度在back propagation的过程中消失，导致训练效果很差，就是我们所熟知的梯度弥散。而深度残差网络在神经网络的结构层面解决了这一问题，使得就算网络很深，梯度也不会消失。下图为深度残差网络的一个基本结构单元。
![b2213d7b1ab12d411e26ad00db458950.png](en-resource://database/1441:1)
我们可以通过深度残差网络的结构可知，深度残差网络的效果即使退化也会比浅层次的神经网络效果要好。
## 4 实验结果
### Dense+ReLU
---
batch_accuracy
![e21b81448fcfa90b9e9babab6f3ac4bc.svg+xml](en-resource://database/1443:1)
batch_loss
![ab6024d5b190c066a540026e77df8be6.svg+xml](en-resource://database/1445:1)
epoch_accuracy
![c19e0041961c68245dbda6f50244e6bb.svg+xml](en-resource://database/1447:1)
epoch_loss![cbfeabf0b932baf1575e5e78ca6125c3.svg+xml](en-resource://database/1449:1)
### LeNet
---
batch_accuracy
![3902e841bc7d6124ea1b9b539507bca5.svg+xml](en-resource://database/1451:1)
batch_loss
![07c49c26db66f6c534d816dd3379fa79.svg+xml](en-resource://database/1453:1)
epoch_accuracy
![d6a5cf8bf5ee3bf88b74f23fc29cb75c.svg+xml](en-resource://database/1455:1)
epoch_loss
![7908bcecad9cbcf64a0e3bd8db3499bb.svg+xml](en-resource://database/1457:1)
### ResNet
---
batch_accuracy
![d444205b055e6891b71f00304bca852a.svg+xml](en-resource://database/1459:1)
batch_loss
![7980cf88129353e9b45e18ae6bcae00e.svg+xml](en-resource://database/1461:1)
epoch_accuracy
![21d9eb9f514f39ea3166fa6dc52be16d.svg+xml](en-resource://database/1463:1)
epoch_loss
![b6a09b0856ccd1f0ca7c272dc8161bfe.svg+xml](en-resource://database/1465:1)
### 结论
在网络深度上ResNet > LeNet > Denset+ReLU，在实现效果上， ResNet > LeNet > Denset+ReLU，我们可以得出网络深度越深时，网络对于Mnist数据集的识别效果越好的结论。三个方法对于Mnist数据集的识别效果在初始迭代的时刻就已经效果达到了较优的值，说明Mnist数据集的特征较为简单，所以利用简单的神经网络结构便可以很有效的进行识别。
