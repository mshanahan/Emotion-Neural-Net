#author: Michael Shanahan 42839964
import model
import util
import tensorflow as tf
import numpy as np
import math

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
  
  #count data
  valid_count = [valid_data[0].shape[0], valid_data[1].shape[0], valid_data[2].shape[0], valid_data[3].shape[0]]
  train_count = [train_data[0].shape[0], train_data[1].shape[0], train_data[2].shape[0], train_data[3].shape[0]]

  #specify model
  input_placeholder = tf.placeholder(tf.float32, [None,16641], name='input_placeholder')
  my_network = tf.identity(model.build_network(input_placeholder),name='output')

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
  
  EPOCHS_BEFORE_STOPPING = 12
  
  #run the actual training
  #code adapted from Paul Quint's hackathon 3
  with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    best_epoch = [0, 0, 0, 0]
    best_valid_ce = [10, 10, 10, 10]
    best_train_ce = [0, 0, 0, 0]
    best_classification_rate = [0, 0, 0, 0]
    epochs_since_best = [0, 0, 0, 0]

    for k in range(0,4):
      session.run(tf.global_variables_initializer())
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
          _, train_ce = session.run([train_op, sum_cross_entropy], {input_placeholder: batch_data, labels: batch_labels})
          ce_vals.append(train_ce)
        avg_train_ce = sum(ce_vals) / len(ce_vals)
        print('TRAIN CROSS ENTROPY: ' + str(avg_train_ce))

        epochs_since_best[k] += 1

        if(best_valid_ce[k] > avg_valid_ce): #tracking best
          best_valid_ce[k] = avg_valid_ce
          best_train_ce[k] = avg_train_ce
          best_epoch[k] = epoch
          best_classification_rate[k] = classification_rate
          epochs_since_best[k] = 0
          print("BEST FOUND")

        if(epochs_since_best[k] >= EPOCHS_BEFORE_STOPPING): #early stopping
          print("EARLY STOP")
          break

        print("\n##################################################")

    print('Avg Best Epoch: ' + str(np.average(best_epoch)))
    print('Avg Valid CE: ' + str(np.average(best_valid_ce)))
    print('Avg Train CE: ' + str(np.average(best_train_ce)))
    print('Avg Classification Rate: ' + str(np.average(best_classification_rate)))
    print('Generating model now...')
    session.run(tf.global_variables_initializer())
    epochs_to_train_for = math.ceil(np.average(best_epoch))

    for j in range(0,4):
      for epoch in range(epochs_to_train_for):
        for i in range(train_count[j] // batch_size):
          batch_data = train_data[j][i*batch_size:(i+1)*batch_size, :]
          batch_labels = train_labels[j][i*batch_size:(i+1)*batch_size]
          _, train_ce = session.run([train_op, sum_cross_entropy], {input_placeholder: batch_data, labels: batch_labels})

    saver.save(session, FLAGS.save_dir)
    print('Model is generated and saved')

if __name__ == "__main__":
    tf.app.run()