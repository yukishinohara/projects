import tensorflow as tf
import numpy as np

WW = np.array([[0.1, 0.6, -0.9], 
               [0.2, 0.5, -0.8], 
               [0.3, 0.4, -0.7],
               [0.4, 0.3, -0.6],
               [0.5, 0.2, -0.5]]).astype(np.float32)
bb = np.array([0.3, 0.4, 0.5]).astype(np.float32)
x_data = np.random.rand(100,5).astype(np.float32)
y_data = np.dot(x_data, WW) + bb

with tf.Session() as sess:

    W = tf.Variable(tf.random_uniform([5,3], -1.0, 1.0))
    # The zeros set to zero with all elements.
    b = tf.Variable(tf.zeros([3]))
    #y = W * x_data + b
    y = tf.matmul(x_data, W) + b

    # Add summary ops to collect data
    w_hist = tf.summary.histogram("weights", W)
    b_hist = tf.summary.histogram("biases", b)
    y_hist = tf.summary.histogram("y", y)

    # Minimize the mean squared errors.
    loss = tf.reduce_mean(tf.square(y - y_data))
    # Outputs a Summary protocol buffer with scalar values
    loss_summary = tf.summary.scalar("loss", loss)

    # Gradient descent algorithm
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

    # Before starting, initialize the variables.  We will 'run' this first.
    init = tf.initialize_all_variables()

    # Creates a SummaryWriter
    # Merges all summaries collected in the default graph
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("tensorflow_log", sess.graph)
    sess.run(init)

    # Fit the line
    for step in range(501):
        if step % 10 == 0:
            result = sess.run([merged, loss])
            summary_str = result[0]
            acc = result[1]
            writer.add_summary(summary_str, step)
            print("step = %s acc = %s W = %s b = %s" % (step, acc, sess.run(W), sess.run(b)))
        else:
            sess.run(train)


