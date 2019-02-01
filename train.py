####
#modified by Yuanwi 2017-12-19
####

import os
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import driving_data
import model

LOGDIR = './save_focal_1_1_11'
#LOGDIR = './save_square'

# add by Yuanwei loss parameters 2018-05-19
alpha_constant = tf.placeholder_with_default(input=1.0, shape=()) 
beta = tf.placeholder_with_default(input=1.0, shape=()) 
alpha = tf.placeholder_with_default(input=1.0, shape=())  
gamma = tf.placeholder_with_default(input=1.1, shape=())
# end loss parameters 2018-05-19


sess = tf.InteractiveSession()

L2NormConst = 0.001

train_vars = tf.trainable_variables()
#modified by Yuanwei 20171215
#loss = tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst

# modify the loss function by Yuanwei 2018-05-19
#loss = tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
#loss = tf.reduce_mean(tf.multiply(alpha , tf.multiply(tf.pow(model.y_, gamma), tf.square(tf.subtract(model.y_, model.y))))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst

loss = tf.reduce_mean(tf.multiply(tf.pow(tf.add(alpha_constant ,tf.multiply(alpha, tf.pow(tf.abs(model.y_), beta))), gamma), tf.square(tf.subtract(model.y_, model.y)))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst

# end by Yuanwei 2018-05-19

#end by Yuanwei 20171215
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
sess.run(tf.initialize_all_variables())

# create a summary to monitor cost tensor
#modified by Yuanwei 20171215
#tf.scalar_summary("loss", loss)
tf.summary.scalar("loss", loss)
#end by Yuanwei 20171215
# merge all summaries into a single op
#modified by Yuanwei 20171215
#merged_summary_op = tf.merge_all_summaries()
merged_summary_op = tf.summary.merge_all()
#end by Yuanwei 20171215
saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V2)

# op to write logs to Tensorboard
#modified by Yuanwei 20171215
logs_path = './logs_focal_1_1_11'
#logs_path = './logs_square'
#summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
#end by Yuanwei 20171215

epochs = 1000
batch_size = 100

loss_value = 0.0
loss_his = 100.0

count_check = 0

loss_value_stop = []
def early_stop(num, loss_value):
  if len(loss_value_stop)<num:
    loss_value_stop.append(loss_value)
    return False
  old_mean_loss = sum(loss_value_stop)/len(loss_value_stop)

  del loss_value_stop[0]
  loss_value_stop.append(loss_value)
  new_mean_loss = sum(loss_value_stop)/len(loss_value_stop)

  print("get the loss_old %g, and the loss_new %g" % (new_mean_loss, old_mean_loss))

  if new_mean_loss>old_mean_loss:
    return True
  else:
    return False


# train over the dataset about 30 times
for epoch in range(epochs):
  for i in range(int(driving_data.num_images/batch_size)):
    xs, ys = driving_data.LoadTrainBatch(batch_size)

    train_step.run(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 0.8})
    
    if i % 10 == 0:
      xs, ys = driving_data.LoadValBatch(batch_size)
      loss_value = loss.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
      print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * batch_size + i, loss_value))

    # write logs at every iteration
    summary = merged_summary_op.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
    summary_writer.add_summary(summary, epoch * driving_data.num_images/batch_size + i)

    # if i % batch_size == 0:
    #   if not os.path.exists(LOGDIR):
    #     os.makedirs(LOGDIR)
    #   checkpoint_path = os.path.join(LOGDIR, "model.ckpt")
    #   filename = saver.save(sess, checkpoint_path)

    if not os.path.exists(LOGDIR):
      os.makedirs(LOGDIR)
    if loss_value < loss_his:
      checkpoint_path = os.path.join(LOGDIR, "model.ckpt")
      filename = saver.save(sess, checkpoint_path)
      print("Best Model saved in file: %s with loss_value %g" % (filename, loss_value))
      loss_his = loss_value

      count_check = 0

    print("count_check:", count_check)

    if count_check>1000:
      print("early stop!")
      exit()

    count_check +=1


    # stop_check = early_stop(5, loss_value)
    # if stop_check:
    #   print("early stop!")
    #   exit()
    # else:
    #   print("do not stop!")

  print("Model saved in file: %s" % filename)

print("Run the command line:\n" \
          "--> tensorboard --logdir=./logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")
