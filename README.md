# How-to-Learn-Tensorflow


DAY 1
=====

一个TensorFlow图由下面详细描述的几个部分组成：

    占位符变量（Placeholder）用来改变图的输入。
    模型变量（Model）将会被优化，使得模型表现得更好。
    模型本质上就是一些数学函数，它根据Placeholder和模型的输入变量来计算一些输出。
    一个cost度量用来指导变量的优化。
    一个优化策略会更新模型的变量。






cuDNN :/usr/bin/ld: 找不到 -lcudnn 
ImportError: cuDNN not available: Can not compile with cuDNN. 
should follow:
 ======================================================================
NVIDIA provides a library for common neural network operations that especially speeds up Convolutional Neural Networks (CNNs). Again, it can be obtained from NVIDIA (after registering as a developer): https://developer.nvidia.com/cudnn

Note that it requires a reasonably modern GPU with Compute Capability 3.0 or higher; see NVIDIA’s list of CUDA GPUs.

To install it, copy the *.h files to /usr/local/cuda/include and the lib* files to /usr/local/cuda/lib64.

To check whether it is found by Theano, run the following command:

python -c "from theano.sandbox.cuda.dnn import dnn_available as d; print(d() or d.msg)"

ImportError: No module named cv2
 ======================================================================
 pip install opencv-python

 
 Ubuntu14.04和16.04官方默认更新源sources.list和第三方源推荐（干货！）
  ======================================================================
  http://www.cnblogs.com/zlslch/p/6860229.html
  
  Caffe 
    ======================================================================
  http://blog.csdn.net/zouyu1746430162/article/details/54095807
  
  http://blog.csdn.net/xierhacker/article/details/53035989
