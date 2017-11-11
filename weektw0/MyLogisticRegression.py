import tensorflow as tf
from numpy import *
def load_dataset():
    X = [[1, float32((xi*xi)%5)] for xi in range(1, 200)]
    Y = [xi%2 for xi in range(1,200)]
    return X, Y

def main():
    x_train,y_train=load_dataset()
    y = y_train
    y_train = mat(y_train)

    theta = tf.Variable(tf.zeros([2, 1]))
    theta0 = tf.Variable(tf.zeros([1, 1]))
    y = 1 / (1 + tf.exp(-tf.matmul(x_train, theta) + theta0))

    loss = tf.reduce_mean(- y_train.reshape(-1, 1) * tf.log(y) - (1 - y_train.reshape(-1, 1)) * tf.log(1 - y))
    train = tf.train.GradientDescentOptimizer(0.005).minimize(loss)#实现梯度下降算法并添加loss来更新来最小化var_list。

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)
    for step in range(1500):
        sess.run(train)
    print(step, sess.run(theta).flatten(), sess.run(theta0).flatten())


if __name__ == '__main__':
    main()