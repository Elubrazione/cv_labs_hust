# 参考资料
## 常见CNN
### VGGNet
在VGG中，每个卷积层后面都**跟着一个ReLU激活函数**，池化层采用的是**最大池化**，而且每隔几个卷积层就会进行一次**降采样**（即池化操作），以减小特征图的大小和复杂度。

VGG网络共有16层或19层，其中16层的VGG被称为VGG16，包括13个卷积层、5个池化层和3个全连接层，而19层的VGG被称为VGG19，包括16个卷积层、5个池化层和3个全连接层。这些层都是以相同的方式排列在网络中，但不同的VGG网络具有不同的层数和参数数量。


### ResNet
ResNet是一种深度残差网络，主要特点是使用**残差块**来构建深层神经网络，解决了深度网络中的**梯度消失和梯度爆炸**问题，同时提高了网络的精度。

在传统的深度神经网络中，由于网络的深度增加，梯度会逐渐变小，导致训练过程中出现梯度消失问题，这会影响网络的精度。为了解决这个问题，ResNet使用残差块来构建网络。残差块中引入了**一个跳跃连接**，使得前向传播中的梯度可以直接传递到后面的层，从而解决了梯度消失问题。

ResNet网络结构中**包含多个残差块**，每个残差块中包含**多个卷积层和批量归一化层**。残差块中的卷积层使用了较**小的卷积核**（通常是3x3），这可以减少网络的参数数量，并提高网络的泛化性能。


### Inception
Inception是谷歌研究团队于2014年提出的一种卷积神经网络结构。与传统的卷积神经网络不同，Inception使用了多**个不同大小的卷积核和池化层**来提取特征，并通过**多个分支将不同尺寸的特征图合并**。这种设计可以提高网络的感受野，并在一定程度上避免了卷积核大小的选择问题，同时也可以减少网络的参数量。

Inception网络结构的核心是**Inception模块**，一个Inception模块包含了多个不同大小的卷积核和池化层。在每个Inception模块中，输入特征图会**分别经过多个卷积层和池化层**进行特征提取，然后再将不同尺寸的**特征图进行合并**，最终输出一个更加丰富的特征图。不同尺寸的卷积核和池化层可以有效地捕捉不同尺度的图像特征，而特征图的合并则可以提高特征的丰富性和复杂度。

除了Inception模块外，Inception网络结构中还包含了多个其他的卷积层、池化层和全连接层。在训练时，Inception网络结构通常使用**dropout和权重衰减**等技术来避免过拟合，并使用**批量归一化**来加速训练和提高网络的精度。