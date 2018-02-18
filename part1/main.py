#author: Michael Shanahan 42839964
import model
import util
import tensorflow as tf
import numpy as np

#code to set flags by Paul Quint
flags = tf.app.flags
flags.DEFINE_string('data_dir', '/work/cse496dl/shared/homework/02/EMODB-German/', 'directory where MNIST is located')
flags.DEFINE_string('save_dir', '/work/cse496dl/mshanaha/homework_2/emodb_homework_2-0-0', 'directory where model graph and weights are saved')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('max_epoch_num', 100, '')
FLAGS = flags.FLAGS

def main(argv):
  #load data
  valid_data_1 = np.load(FLAGS.data_dir + 'test_x_1.npy')
  valid_data_2 = np.load(FLAGS.data_dir + 'test_x_2.npy')
  valid_data_3 = np.load(FLAGS.data_dir + 'test_x_3.npy')
  valid_data_4 = np.load(FLAGS.data_dir + 'test_x_4.npy')
  valid_data = [valid_data_1, valid_data_2, valid_data_3, valid_data_4]
  
  valid_labels_1 = np.load(FLAGS.data_dir + 'test_y_1.npy')
  valid_labels_2 = np.load(FLAGS.data_dir + 'test_y_2.npy')
  valid_labels_3 = np.load(FLAGS.data_dir + 'test_y_3.npy')
  valid_labels_4 = np.load(FLAGS.data_dir + 'test_y_4.npy')
  valid_labels = [valid_labels_1, valid_labels_2, valid_labels_3, valid_labels_4]
  
  train_data_1 = np.load(FLAGS.data_dir + 'train_x_1.npy')
  train_data_2 = np.load(FLAGS.data_dir + 'train_x_2.npy')
  train_data_3 = np.load(FLAGS.data_dir + 'train_x_3.npy')
  train_data_4 = np.load(FLAGS.data_dir + 'train_x_4.npy')
  train_data = [train_data_1, train_data_2, train_data_3, train_data_4]
  
  train_labels_1 = np.load(FLAGS.data_dir + 'train_y_1.npy')
  train_labels_2 = np.load(FLAGS.data_dir + 'train_y_2.npy')
  train_labels_3 = np.load(FLAGS.data_dir + 'train_y_3.npy')
  train_labels_4 = np.load(FLAGS.data_dir + 'train_y_4.npy')
  train_labels = [train_labels_1, train_labels_2, train_labels_3, train_labels_4]
  
#  print(str(valid_data_1[0*32:(0+1)*32, :]))
#  print(str(valid_labels_1[0*32:(0+1)*32, :]))
  
  #count data
  valid_count = [valid_data[0].shape[0], valid_data[1].shape[0], valid_data[2].shape[0], valid_data[3].shape[0]]
  train_count = [train_data[0].shape[0], train_data[1].shape[0], train_data[2].shape[0], train_data[3].shape[0]]

  #specify model
  input_placeholder = tf.placeholder(tf.float32, [None,16641], name='input_placeholder')
  conv_layer = model.convolutional_layer(input_placeholder)
  linear_layer = model.linear_layer(conv_layer)
  output_layer = tf.layers.dense(linear_layer, 7, name = 'output_layer')
  my_network = tf.identity(output_layer,name='output')

  #define classification loss
  #code adapted from Paul Quint's hackathon 3
  REG_COEFF = 0.0001
  labels = tf.placeholder(tf.float32, [None, 7], name='labels')
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=my_network)
  confusion_matrix_op = tf.confusion_matrix(tf.argmax(labels, axis=1), tf.argmax(my_network, axis=1), num_classes=7)
  regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  total_loss = cross_entropy + REG_COEFF * sum(regularization_losses)

  #set up training and saving
  #code adapted from Paul Quint's hackathon 3
  global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
  optimizer = tf.train.AdamOptimizer()
  train_op = optimizer.minimize(total_loss, global_step=global_step_tensor)
  saver = tf.train.Saver()
  sum_cross_entropy = tf.reduce_mean(cross_entropy)
  
  best_k = 5
  best_all_epoch = 0
  best_all_valid_ce = 10
  best_all_train_ce = 0
  best_all_classification_rate = 0
  EPOCHS_BEFORE_STOPPING = 12
  
  #run the actual training
  #code adapted from Paul Quint's hackathon 3
  with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for k in range(0,4):
      best_epoch = 0
      best_valid_ce = 10
      best_train_ce = 0
      best_classification_rate = 0
      epochs_since_best = 0
      batch_size = FLAGS.batch_size
      print("\n !!!!! NEW K (" + str(k) + ") !!!!!\n")
      for epoch in range(FLAGS.max_epoch_num):
        print("################### EPOCH " + str(epoch) + " #####################")
        print("##################################################\n")
        
        # run gradient steps and report mean loss on train data
        ce_vals = []
        conf_mxs = []
        for i in range(valid_count[k] // batch_size):
          batch_data = valid_data[k][i*batch_size:(i+1)*batch_size, :]
          batch_labels = valid_labels[k][i*batch_size:(i+1)*batch_size]
          #batch_labels = util.white_hot(batch_labels)
          valid_ce, conf_matrix = session.run([sum_cross_entropy, confusion_matrix_op], {input_placeholder: batch_data, labels: batch_labels})
          ce_vals.append(valid_ce)
          conf_mxs.append(conf_matrix)
        avg_valid_ce = sum(ce_vals) / len(ce_vals)
        print('VALID CROSS ENTROPY: ' + str(avg_valid_ce))
        print('VALIDATION CONFUSION MATRIX:')
        print(str(sum(conf_mxs)))
        classification_rate = util.classification_rate(sum(conf_mxs),7)
        print('VALIDATION CLASSIFICATION RATE:' + str(classification_rate))
        
        ce_vals = []
        for i in range(train_count[k] // batch_size):
          batch_data = train_data[k][i*batch_size:(i+1)*batch_size, :]
          batch_labels = train_labels[k][i*batch_size:(i+1)*batch_size]
          #batch_labels = util.white_hot(batch_labels)
          _, train_ce = session.run([train_op, sum_cross_entropy], {input_placeholder: batch_data, labels: batch_labels})
          ce_vals.append(train_ce)
        avg_train_ce = sum(ce_vals) / len(ce_vals)
        print('TRAIN CROSS ENTROPY: ' + str(avg_train_ce))
        
        epochs_since_best += 1
        
        if(best_valid_ce > avg_valid_ce): #tracking best
          best_valid_ce = avg_valid_ce
          best_train_ce = avg_train_ce
          best_epoch = epoch
          best_classification_rate = classification_rate
          epochs_since_best = 0
          print("BEST FOUND")
          
        if(best_all_valid_ce > best_valid_ce): #tracking best of all time
          best_all_valid_ce = best_valid_ce
          best_all_train_ce = best_train_ce
          best_all_epoch = best_epoch
          best_all_classification_rate = best_classification_rate
          best_k = k
          saver.save(session, FLAGS.save_dir)
          print("BEST ALL TIME FOUND")

        if(epochs_since_best >= EPOCHS_BEFORE_STOPPING): #early stopping
          print("EARLY STOP")
          break
          
        print("\n##################################################")
        
      print('Best Epoch: ' + str(best_epoch))
    print('Best k: ' + str(best_k))
    print('Best epoch of k: ' + str(best_all_epoch))

if __name__ == "__main__":
    tf.app.run()
