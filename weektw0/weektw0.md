#WEEKTW0
##理论问题：

- **1  back propagation 算法原理理解 ？**
	- BackPropagation 反向传播算法,即常说的BP算法。
	- 一般的算法是自下而上的计算路径权值：
	![](https://i.imgur.com/rriGdb5.png)
	- 但是BP算法正好相反，是自上而下的计算权值，这样的好处有：
		- 避免了一些路径的重复使用
	- 如：
	![](https://i.imgur.com/HipWljf.png)
	- 当自下而上的计算权值的时候，a-c-e和b-c-e就都走了路径c-e，当对于权值极大的深度模型中的神经网络，这样的冗余所导致的计算量是相当大的。
	
- **2  sigmoid函数、tanh函数和ReLU函数的区别？以及各⾃的优缺点？对应的
tf函数是？** 
 - 参考： http://cs231n.github.io/neural-networks-1/#actfun
![](https://i.imgur.com/rkPzuYn.png)
	- **Sigmoid**. The sigmoid non-linearity has the mathematical form σ(x)=1/(1+e−x)σ(x)=1/(1+e−x) and is shown in the image above on the left. As alluded to in the previous section, it takes a real-valued number and “squashes” it into range between 0 and 1. In particular, large negative numbers become 0 and large positive numbers become 1. The sigmoid function has seen frequent use historically since it has a nice interpretation as the firing rate of a neuron: from not firing at all (0) to fully-saturated firing at an assumed maximum frequency (1). In practice, the sigmoid non-linearity has recently fallen out of favor and it is rarely ever used. It has two major drawbacks:
	
	> sigmoid非线性激活函数,它将实数压缩到0 1 区间。而且，大的负数变成0，大的正数变成1。sigmoid函数由于其强大的解释力，在历史上被最经常地用来表示神经元的活跃度：从不活跃（0）到假设上最大的（1）。在实践中，sigmoid函数最近从受欢迎到不受欢迎，很少再被使用。它有两个主要缺点
	
 	- Sigmoids saturate and kill gradients. A very undesirable property of the sigmoid neuron is that when the neuron’s activation saturates at either tail of 0 or 1, the gradient at these regions is almost zero. Recall that during backpropagation, this (local) gradient will be multiplied to the gradient of this gate’s output for the whole objective. Therefore, if the local gradient is very small, it will effectively “kill” the gradient and almost no signal will flow through the neuron to its weights and recursively to its data. Additionally, one must pay extra caution when initializing the weights of sigmoid neurons to prevent saturation. For example, if the initial weights are too large then most neurons would become saturated and the network will barely learn.
 	> sigmoid过饱和、丢失了梯度。sigmoid神经元的一个很差的属性就是神经元的活跃度在0和1处饱和，它的梯度在这些地方接近于0。回忆在反向传播中，某处的梯度和其目标输出的梯度相乘，以得到整个目标。因此，如果某处的梯度过小，就会很大程度上干掉梯度，使得几乎没有信号经过这个神经元以及所有间接经过此处的数据。除此之外，人们必须额外注意sigmoid神经元权值的初始化来避免饱和。例如，当初始权值过大，几乎所有的神经元都会饱和以至于网络几乎不能学习。
	- Sigmoid outputs are not zero-centered. This is undesirable since neurons in later layers of processing in a Neural Network (more on this soon) would be receiving data that is not zero-centered. This has implications on the dynamics during gradient descent, because if the data coming into a neuron is always positive (e.g. x>0x>0 elementwise in f=wTx+bf=wTx+b)), then the gradient on the weights ww will during backpropagation become either all be positive, or all negative (depending on the gradient of the whole expression ff). This could introduce undesirable zig-zagging dynamics in the gradient updates for the weights. However, notice that once these gradients are added up across a batch of data the final update for the weights can have variable signs, somewhat mitigating this issue. Therefore, this is an inconvenience but it has less severe consequences compared to the saturated activation problem above.
	> sigmoid的输出不是零中心的。这个特性会导致为在后面神经网络的高层处理中收到不是零中心的数据。这将导致梯度下降时的晃动，因为如果数据到了神经元永远时正数时，反向传播时权值w就会全为正数或者负数。这将导致梯度下降不希望遇到的锯齿形欢动。但是，如果采用这些梯度是由批数据累加起来，最终权值更新时就会更准确。因此，这是一个麻烦一些，但是能比上面饱和的激活问题结果好那么一些。
	- **Tanh**. The tanh non-linearity is shown on the image above on the right. It squashes a real-valued number to the range [-1, 1]. Like the sigmoid neuron, its activations saturate, but unlike the sigmoid neuron its output is zero-centered. Therefore, in practice the tanh non-linearity is always preferred to the sigmoid nonlinearity. Also note that the tanh neuron is simply a scaled sigmoid neuron, in particular the following holds: tanh(x)=2σ(2x)−1tanh⁡(x)=2σ(2x)−1.
	> 将实数映射到[-1,1]。就像sigmoid神经元一样，它的激活也会好合，但是不像sigmoid函数，它是零中心的。因此在实际应用中tanh比sigmoid更优先使用。而且注意到tanh其实很简单，是sigmoid缩放版。
	- **ReLU**. The Rectified Linear Unit has become very popular in the last few years. It computes the function f(x)=max(0,x)f(x)=max(0,x). In other words, the activation is simply thresholded at zero (see image above on the left). There are several pros and cons to using the ReLUs:
	> ReLU近几年很流行。它求函数f(x) = max(0,x)。换句话说，这个激活函数简单设0为阈值（见上图左—）。关于用ReLU有这些好处和坏处：

	* (+) It was found to greatly accelerate (e.g. a factor of 6 in Krizhevsky et al.) the convergence of stochastic gradient descent compared to the sigmoid/tanh functions. It is argued that this is due to its linear, non-saturating form.
	> 其在梯度下降上比较tanh/sigmoid有更快的收敛速度。这被认为时其线性、非饱和的形式。
	* (+) Compared to tanh/sigmoid neurons that involve expensive operations (exponentials, etc.), the ReLU can be implemented by simply thresholding a matrix of activations at zero.
	> 比较tanh/sigmoid操作开销大（指数型），ReLU可以简单设计矩阵在0的阈值来实现。
	* (-) Unfortunately, ReLU units can be fragile during training and can “die”. For example, a large gradient flowing through a ReLU neuron could cause the weights to update in such a way that the neuron will never activate on any datapoint again. If this happens, then the gradient flowing through the unit will forever be zero from that point on. That is, the ReLU units can irreversibly die during training since they can get knocked off the data manifold. For example, you may find that as much as 40% of your network can be “dead” (i.e. neurons that never activate across the entire training dataset) if the learning rate is set too high. With a proper setting of the learning rate this is less frequently an issue.
	> 不幸的是，ReLU单元脆弱且可能会在训练中死去。例如，大的梯度流经过ReLU单元时可能导致神经不会在以后任何数据节点再被激活。当这发生时，经过此单元的梯度将永远为零。ReLU单元可能不可逆地在训练中的数据流中关闭。例如，比可能会发现当学习速率过快时你40%的网络都“挂了”（神经元在此后的整个训练中都不激活）。当学习率设定恰当时，这种事情会更少出现。
	
  - **sigmoid函数 -> tf.sigmoid(x, name=None)**
  - **tanh函数    -> tf.tanh(x, name=None)**
  - **ReLU函数    -> tf.nn.relu6(features, name=None)**
  - 部分译文来自：http://blog.csdn.net/csuhoward/article/details/52538296
##3  softmax和cross_entropy原理解释？
  - **1 softmax原理理解：**
  	* **1.1我的理解**
	  	* softmax = soft + max，那么什么是max？a>b时，max（a,b）=a。也就是说在a大于b时，max(a,b)=a绝对成立，但这样就造成了较小值的饥饿。所以当我希望数值大的那个可以经常取到，同时数值小的那个也可以被偶尔取到，那么softmax就可以了。
  	* **1.2定义**
	  	* 对于拥有i个元素的数组V，我们用Vi表示其中第i个元素，那么这个元素的softmax的值就是  
	  	![](https://i.imgur.com/S9mYPti.png)  
		也就是{e的vi次幂，与【（任意属于V的Vj）所对应e的Vj次幂】的和}的比值  
![](https://i.imgur.com/9N8ZrRE.png)
  - **2 cross_entropy原理理解**  
	* cross_entropy 交叉熵。熵在信息论中的定义是一个信息具有的信息量。  
	    ![](https://i.imgur.com/72HPXo9.png)
	* KL散度表示从消息A的角度来看，消息B有多大不同，用于计算代价  
		交叉熵表示消息A的角度看，如何描述消息B。 **在特定的情况下最小化KL散度等价于最小化交叉熵** 
	* 对于离散事件，我们可以定义事件A.B的KL散度为：
		![](https://i.imgur.com/YrU4Nck.png)
	由上式可知，
		* 如果PA = PB，即两个事件分布完全相同，即KL散度为0
		* 第二个等号右侧，减号左边是事件A的熵
	* 对于离散事件，我们可以定义事件A,B的交叉熵公式：
		![](https://i.imgur.com/465NVbH.png)  
	* 综合以上两个公式，我们发现 KL散度= 交叉熵 - 熵，所以上面说到的特定条件即为，如果熵S(A)为常量，则Dkl(A||B)~H(A,B),即在特定条件下等价。
	* 我们所用来训练的数据A的分布是给定的，也就是说求Dkl(A||B)等价与求H(A,B)，所以cross\_entropy可以用于计算“学习模型的分布”和“训练数据分布”之间的不同，当cross_entropy最小时，我们得到了“最好的模型”
  - **3 关联**
  	* 3.1 交叉熵损失函数
	  	* 定义  
	  ![](https://i.imgur.com/cZJy9Dm.png)
	* 3.2 关系
		* 当激活函数为sigmoid函数时，会导致参数更新缓慢，如果选择普通均方误差损失函数，会遇到上述问题（参数更新缓慢）。但交叉熵损失函数则可以克服这种问题。
		* 所以交叉熵损失函数的导数具有如下形式：  
		![](https://i.imgur.com/Yz35Jbi.png)  
		没有了一阶导项，收敛结果更好
  - **参考**  
	  - 1 Softmax 函数的特点和作用是什么？ - 杨思达zzzz的回答 - 知乎
		https://www.zhihu.com/question/23765351/answer/139826397
	  - 2 为什么交叉熵（cross-entropy）可以用于计算代价？ - 阿萨姆的回答 - 知乎
		https://www.zhihu.com/question/65288314/answer/244557337
	  - 3 http://blog.csdn.net/u012494820/article/details/52797916
##4  tf.placeholder() 、tf.constant()、tf.Variable()的区别？
  - tf.Variable():
  	- Variable()构造函数需要变量的初始值，它可以是任何类型和形状的张量。 初始值定义变量的类型和形状。 构造后，变量的类型和形状是固定的。 该值可以使用其中一种赋值方法进行更改
	  > The Variable() constructor requires an initial value for the variable, which can be a Tensor of any type and shape. The initial value defines the type and shape of the variable. After construction, the type and shape of the variable are fixed. The value can be changed using one of the assign methods.
  - tf.placeholder()
    - 用于得到传递进来的真实的训练样本。  
		不必指定初始值，可在运行时，通过 Session.run 的函数的 feed_dict 参数指定；
		这也是其命名的原因所在，仅仅作为一种占位符； 用于执行之前必须赋值。
	  > Inserts a placeholder for a tensor that will be always fed.  
  - tf.constant()
    - 创建一个常数张量。  
	  得到的张量由参数值和（可选）形状（参见下面的示例）指定的类型为dtype的值填充。  
	  参数值可以是一个常量值，也可以是一个dtype类型的值列表。 如果value是一个列表，那么列表的长度必须小于或等于shape参数所暗示的元素的数量（如果指定的话）。 在列表长度小于shape指定的元素数量的情况下，列表中的最后一个元素将被用于填充剩余的条目。  
      参数形状是可选的。 如果存在，则指定所得张量的尺寸。 如果不存在，则如果值是标量，则张量是标量（0-D），否则是张量。  
      如果没有指定参数dtype，那么类型是从值类型推断的。
  	 > Creates a constant tensor.  
  	 > The resulting tensor is populated with values of type dtype, as specified by arguments value and (optionally) shape (see examples below).  
  	 > The argument value can be a constant value, or a list of values of type dtype. If value is a list, then the length of the list must be less than or equal to the number of elements implied by the shape argument (if specified). In the case where the list length is less than the number of elements specified by shape, the last element in the list will be used to fill the remaining entries.
  	 > The argument shape is optional. If present, it specifies the dimensions of the resulting tensor. If not present, then the tensor is a scalar (0-D) if value is a scalar, or 1-D otherwise.
  	 > If the argument dtype is not specified, then the type is inferred from the type of value.
##5  举例说明：tf.Graph() 概念理解？
  - class tf.Graph  
  	A TensorFlow computation, represented as a dataflow graph.  
	A Graph contains a set of Operation objects, which represent units of computation; and Tensor objects, which represent the units of data that flow between operations.  
	A default Graph is always registered, and accessible by calling tf.get_default_graph(). To add an operation to the default graph, simply call one of the functions that defines a new Operation:
	> c = tf.constant(4.0)  
	> assert c.graph is tf.get_default_graph() 
 
	**Important note** This class is not thread-safe for graph construction. All operations should be created from a single thread, or external synchronization must be provided. Unless otherwise specified, all methods are not thread-safe.
  - 以上是tensorflow官网的上的定义。我们可以这样理解，Graph是一个计算单元操作的对象和一个操作中数据流动的张量对象
    > import tensorflow as tf  
    > g1=tf.Graph()  
    > with g1.as_default():  
    > &emsp;&emsp;a = tf.constant(1)  
    > &emsp;&emsp;b = tf.constant(2)  
    > with tf.Graph().as_default() as g2:  
    > &emsp;&emsp;c = tf.constant(3)  
    > &emsp;&emsp;d = tf.constant(4)  
    > with tf.Session(graph=g1) as sess1:  
    >&emsp;&emsp;print sess1.run(a+b)  
    > with tf.Session(graph=g2) as sess2:  
    > &emsp;&emsp;print sess2.run(c+d)  

    运行结果是3，7
	但是如果运行这段代码：
	> with tf.Session(graph=g2) as sess2:  
	>&emsp;&emsp;print sess2.run(a+b)
	
	则会报错，因为a,b是在g1中，而不是在g2中 

##6  tf.name\_scope()和tf.variable\_scope()的理解？
  - tf.variable\_scope()可以同时对variable和ops的命名有影响，即加前缀；而tf.name\_scope()只能对ops的命名加前缀
  - tf.name\_scope():
	  - 参考：https://www.tensorflow.org/api_docs/python/tf/name_scope
	  - A context manager for use when defining a Python op.This context manager validates that the given values are from the same graph, makes that graph the default graph, and pushes a name scope in that graph 
	  - 当命名域重名的时候，tf.name_scope会自动对重名的域打上序号

  - tf.variable\_scope():
	  - 参考： https://www.tensorflow.org/api_docs/python/tf/variable_scope
	  - A context manager for defining ops that creates variables (layers).This context manager validates that the (optional) values are from the same graph, ensures that graph is the default graph, and pushes a name scope and a variable scope.If name_or_scope is not None, it is used as is. If scope is None, then default_name is used. In that case, if the same name has been previously used in the same scope, it will be made unique by appending _N to it.Variable scope allows you to create new variables and to share already created ones while providing checks to not create or share by accident. For details, see the Variable Scope How To, here we present only a few basic examples.
	  - 这个函数会同时创建一个name\_scope，即相当于先调用了一次tf.name\_scope
	  - 如果给定了name\_or\_scope，那么两个重名的变量域对tf.get\_variable生成的变量来说实际上是同一个域，而对于ops来说则是不同的命名域。可以理解为除了tf.get\_variable生成的变量，其他ops均受到这个函数创建的tf.name_scope影响
	  - 如果name_or_scope为None，那么tf将对重复的命名域打上序号，这个对一些函数式的layer很重要
##7  tf.variable\_scope() 和tf.get\_variable()的理解？
  -  我们可以通过tf.Variable()来新建变量，但是，在tensorflow程序中，我们又需要共享变量（share variables），于是，就有了tf.get\_variable()(新建变量或者取已经存在的变量)。但是，因为一般变量命名比较短，那么，此时，我们就需要类似目录工作一样的东西来管理变量命名，于是，就有了tf.variable\_scope()，同时，设置reuse标志，就可以来决定tf.get\_variable()的工作方式（新建变量或者取得已经存在变量）
  -  tf.get\_variable()是新建变量还是取得已存在变量取决于调用它的scope
	  -  情况1：若scope设置为新建变量，例如，tf.get\_variable\_scope().reuse == False.(说明：tf.get\_variable\_scope()得到当前的scope)  
&emsp;&emsp;在这种情况下，v就是一个给定shape和initializer的新建的tf.Variable。新建变量的full name则就是scope的name+tf.get\_variable()函数中的name，同时，还会检查是否已经存在这个full name的变量。如果存在这个full name 的变量，函数会报ValueError错误.
	  - 情况2：若scope设置为重新使用已有变量，例如，tf.get\_variable\_scope().reuse == True.   
&emsp;&emsp;在这种情况下，这个调用则会寻找full name为 scope的name + tf.get_variable()函数中的name的变量。如果不存在，则会报ValueError错误。如果存在，则会返回这个变量。
  - tf.variable\_scope()的基本作用主要是给变量名加前缀和设置reuse标志以此来区分tf.get\_variable()的两种情况（新建变量或者重新使用已有变量）。加前缀类似于目录的工作。
    - 注意：
    	- 1.不要显示的设置reuse=False，reuse的取值为None和True.默认reuse=None,即不共享参数。当reuse=True时，共享参数，并且，在此scope下的所有sub-scope的reuse都为True。
    	- 2.tf.variable\_scope(name_or_scope,…)中的name\_or\_scope可以是VariableScope Object（一般用于较复杂的情况）。
    	- 3.使用之前已经存在的scope来打开一个variable scope，能够跳出当前variable scope前缀而为一个完全不同的scope.无论什么时候这样都完全独立。
    	- 4.在一个区域内，可以为所有的variables设置默认的initializer。设置默认的initializer能够被sub-scope继承，且会传送给同区域的每一个tf.get\_variable()。可以通过显示指定另一个initializer来重写initializer。
  - 参考：http://blog.csdn.net/IB_H20/article/details/72936574?locationNum=8&fps=1


##8  tf.global\_variables\_initializer() 什么时候使用？
  - tf.global\_variables\_initializer()会返回一个操作，从计算图中初始化所有TensorFlow变量。
  - Returns an Op that initializes global variables.  
This is just a shortcut for variables\_initializer(global\_variables())
#实践任务

##1 使用tf实现Logistic Regression算法（必做）
  - 参考：     
	 http://blog.csdn.net/cyh_24/article/details/50359055  
     http://blog.csdn.net/lizhe\_dashuju/article/details/49864569
##2 使用a任务实现的算法，完成 “Kaggle泰坦尼克之灾”（链接https://www.kaggle.com/c/titanic）