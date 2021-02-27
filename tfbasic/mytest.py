import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

with tf.compat.v1.Session() as sess:
    train_step = 30
    for step in range(train_step):
        fl = tf.compat.v1.to_float(step)
        print("{},{}".format(fl.eval(),sess.run(tf.sigmoid(-fl))))