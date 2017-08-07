# How-to-Learn-Tensorflow


DAY 1
=====

一个TensorFlow图由下面详细描述的几个部分组成：

    占位符变量（Placeholder）用来改变图的输入。
    模型变量（Model）将会被优化，使得模型表现得更好。
    模型本质上就是一些数学函数，它根据Placeholder和模型的输入变量来计算一些输出。
    一个cost度量用来指导变量的优化。
    一个优化策略会更新模型的变量。


DAY 2
=====
1、对于输入层（图像层），我们一般会把图像大小resize成边长为2的次方的正方形。比如CIFAR-10是32x32x3，STL-10是64x64x3，而ImageNet是224x224x3或者512x512x3。

2、实际工程中，我们得预估一下内存，然后根据内存的情况去设定合理的值。例如输入是224x224x3得图片，过滤器大小为3x3，共64个，zero-padding为1，这样每张图片需要72MB的内存（这里的72MB囊括了图片以及对应的参数、梯度和激活值在内的，所需要的内存空间），但是在GPU上运行的话，内存可能不够（相比于CPU，GPU的内存要小得多），所以需要调整下参数，比如过滤器大小改为7x7，stride改为2（ZF net），或者过滤器大小改为11x11，stride改为4（AlexNet）。

3、构建一个实际可用的深度卷积神经网络最大的瓶颈是GPU的内(显)存。现在很多GPU只有3/4/6GB的内存，单卡最大的也就12G（NVIDIA），所以我们应该在设计卷积神经网的时候，多加考虑内存主要消耗在哪里：

大量的激活值和中间梯度值；
参数，反向传播时的梯度以及使用momentum，Adagrad，or RMSProp时的缓存都会占用储存，所以估计参数占用的内存时，一般至少要乘以3倍；
数据的batch以及其他的类似信息或者来源信息等也会消耗一部分内存。

作者：Deeplayer
链接：http://www.jianshu.com/p/9c4396653324
來源：简书
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

DAY 3
=====
Read image files
In [3]:

tf.reset_default_graph()
​
# Hyperparams
batch_size = 10
num_epochs = 1
​
# Make fake images and save
for i in range(100):
    _x = np.random.randint(0, 256, size=(10, 10, 4))
    plt.imsave("example/image_{}.jpg".format(i), _x)
​
# Import jpg files
images = tf.train.match_filenames_once('example/*.jpg')
​
# Create a string queue
fname_q = tf.train.string_input_producer(images, num_epochs=num_epochs, shuffle=True)
​
# Q10. Create a WholeFileReader
reader = tf.WholeFileReader()
​
# Read the string queue
_, value = reader.read(fname_q)
​
# Q11. Decode value
img = tf.image.decode_image(value)
​
# Batching
img_batch = tf.train.batch([img], shapes=([10, 10, 4]), batch_size=batch_size)
​
with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
​
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    num_samples = 0
    try:
        while not coord.should_stop():
            sess.run(img_batch)
            num_samples += batch_size
            print(num_samples, "samples have been seen")
​
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
​
    coord.join(threads)
