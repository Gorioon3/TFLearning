一个真·萌新的学习笔记

github https://github.com/Gorioon3/TFLearning
#week on3

##1 Tensorflow安装
 - 环境：Ubuntu  16.04
 - 版本：tensorflow-0.5.0-cp27-none-linux_x86_64（CPU ONLY）
 - 参考文章：http://www.tensorfly.cn/tfdoc/get_started/os_setup.html
 - 踩到的坑：
   - 1 python pip 未安装：
        sudo apt-get install python-pip
   - 2 按照上述方法安装的pip出现版本过低问题：
     * 1.sudo apt-get update
     * 2.sudo apt-get upgrade
     * 3.wget https://bootstrap.pypa.io/get-pip.py  --no-check-certificate
     * 4.sudo python get-pip.py
  - 3 pip install 超时：
     * 下载到本地后，pip install +路径名+包名
 - 一点注意：开始的时候使用的时Ubuntu 17 默认为python3，我拿到的版本里面没有python2.7，需要重新下载安装，当时手贱的把python3卸载了，后果有点可怕。
##2 Tensorflow 初试
   - 打开一个 python 终端:
*$ python 
&gt;&gt;&gt; import tensorflow as tf
&gt;&gt;&gt;hello = tf.constant('Hello, TensorFlow!')
&gt;&gt;&gt; sess = tf.Session()
&gt;&gt;&gt; print sess.run(hello)
Hello, TensorFlow!
&gt;&gt;&gt; a = tf.constant(10)
&gt;&gt;&gt;b = tf.constant(32)
&gt;&gt;&gt; print sess.run(a+b)
42

##3 实践任务：
- 1 已知x =2 ,y =3 , z =7 ,求解 res = x*y+z的解：
 * res = 13。
- 2 矩阵乘法：A:[[3.,3.]]  , B[[2.],[2.]] ,A矩阵和B矩阵的乘法运算。
 * res = [[ 12.]]

##4 思考
- 1、机器学习中，监督学习 or 非监督学习概念区分，应用场景调研？
 * 监督学习 Supervised Learning
       &emsp;&emsp;给出一个算法，需要部分数据集已经有正确答案。比如给出给定房价数据集，对于里面每个数据，算法都能计算出对应的正确房价。算法的结果就是短处更多的正确价格。
 * 非监督学习  Unsupervised Learning
&emsp;&emsp;在非监督学习中，我们可以在很少或者根本不知道我们的输出结果是什么的时候，就可以从数据中得到结构，我们可以基于数据之间的关系，对数据进行聚类，从而获取这种目标结构。在监督学习中，没有基于预测结果的反馈。    
- 2、做机器学习项目，有哪些环节？
 * 目前机器学习经验为 0 （T^T）
但是，在一些大佬的博客上总结的大概是：
<img src="/uploads/default/original/1X/a82bcb906309421a81478d9dfe3590d458dfcc70.png" width="690" height="455">
图源：http://blog.csdn.net/wangyaninglm/article/details/78311126
##3、深度学习，目前有哪些应用领域？
 * 1 Image Classification
 * 2 Speech Recognition
 * 3 Machine Translation
 * 4 精准广告推荐
##4、数据预处理，需要注意哪些？
 * 1 缺失点
 * 2 孤立点
 * 3 垃圾值
 * 4 重复的积累
##5、Tensorflow运型原理，架构有哪些核心点？
- TensorFlow的系统结构以C API为界，将整个系统分为「前端」和「后端」两个子系统：

&emsp;&emsp;前端系统：提供编程模型，负责构造计算图；
&emsp;&emsp;后端系统：提供运行时环境，负责执行计算图。
<img src="/uploads/default/original/1X/32ae779496ac8c24b8cc48c2a30bc8d13ef0a0f0.png" width="629" height="366">
图源：http://www.jianshu.com/p/a5574ebcdeab