import tensorflow as tf
#生成一个绘画来启动图
sess = tf.Session()
#产生一个 1x2 矩阵 并放入图中
matrix1 = tf.constant([[3., 3.]])
#产生一个 2x1 矩阵 并放入图中
matrix2 = tf.constant([[2.],[2.]])
#计算两矩阵乘法
#方法1：
res = sess.run(matrix1*matrix1)
#结果 
#print res  
#[[ 6.  6.]
#[ 6.  6.]]
#方法2:
res = tf.matmul(matrix1, matrix2)
print sess.run(res)
#结果[[ 12.]]
#由此可见方法2 是正确的，原因就是矩阵乘法和向量点乘的区别
sess.close()
