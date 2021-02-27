import tensorflow as tf

tf.compat.v1.disable_v2_behavior()

W = tf.Variable(tf.zeros([2,1]),name="weights")
b = tf.Variable(0.,name="bias")

def inference(X):  #假设，模型的关键
    return tf.compat.v1.matmul(X,W) + b

def loss(X,Y):
    y_predicted = inference(X)
    return tf.compat.v1.reduce_sum(tf.compat.v1.squared_difference(Y,y_predicted))

def train(total_loss):
    learning_rate = 0.00000001
    return tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

def evaluate(sess):
    print("value:",sess.run(inference([[80.,25.]])))

def inputs():
    weight_age = [[84,46],[73,20],[65,52],[70,30],[76,57],[69,25],[63,28],
                  [72,36],[63,28],[72,36],[79,57],[75,44],
                  [27,24],[89,31],[65,52],[57,23],[59,60],
                  [69,48],[60,34],[79,51],[75,50],[82,34],[59,46],[67,23],
                  [85,37],[55,40],[63,30]]
    blood_fat_content = [354,190,405,263,451,302,288,
                         385,402,365,209,290,346,254,395,434,220,
                         374,308,220,311,181,274,303,244]
    return tf.compat.v1.to_float(weight_age),tf.compat.v1.to_float(blood_fat_content)


with tf.compat.v1.Session() as sess:
    tf.compat.v1.initialize_all_variables().run()
    X,Y = inputs()
    total_loss = loss(X,Y)
    train_op = train(total_loss)
    coord = tf.compat.v1.train.Coordinator
    training_steps = 100000
    for step in range(training_steps):
        sess.run([train_op])
        if step %1000 == 0:
            print("loss:",sess.run([total_loss]))
    print(W.eval())
    # coord.request_stop()
    # coord.join(threads=threads)
    sess.close()



