import tensorflow as tf
import numpy as np
import os
import util
import model

# flags
flags = tf.app.flags
flags.DEFINE_string('data_dir', '/work/cse496dl/shared/homework/02/EMODB-German/', 'directory where FMNIST is located')
flags.DEFINE_string('save_dir', '/work/cse496dl/cpack/Assignment_2/models/1', 'directory where model graph and weights are saved')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('max_epoch_num', 2, '')
flags.DEFINE_integer('tol_epoch_num', 2, '')
flags.DEFINE_float('dropout', 0.5, '')
flags.DEFINE_bool('is_testing', False, '')
FLAGS = flags.FLAGS

def main(argv):
    images_1 = np.load(FLAGS.data_dir + 'train_x_1.npy')
    images_2 = np.load(FLAGS.data_dir + 'train_x_2.npy')
    images_3 = np.load(FLAGS.data_dir + 'train_x_3.npy')
    images_4 = np.load(FLAGS.data_dir + 'train_x_4.npy')
    train_images = [images_1, images_2, images_3, images_4]
    
    labels_1 = np.load(FLAGS.data_dir + 'train_y_1.npy')
    labels_2 = np.load(FLAGS.data_dir + 'train_y_2.npy')
    labels_3 = np.load(FLAGS.data_dir + 'train_y_3.npy')
    labels_4 = np.load(FLAGS.data_dir + 'train_y_4.npy')
    train_labels = [labels_1, labels_2, labels_3, labels_4]

    # Reshaping
    #images = np.reshape(images,[-1,129,129,1])         

    #train_images, test_images = split_data(images, 0.9)
    #train_labels, test_labels = split_data(labels, 0.9)
    # Not sure this randomized splitting is acceptable with the use of the k-fold validation.
    #train_images, validation_images = util.split_data(images, 0.9)
    #train_labels, validation_labels = util.split_data(labels, 0.9)
    
    
    images_1 = np.load(FLAGS.data_dir + 'test_x_1.npy')
    images_2 = np.load(FLAGS.data_dir + 'test_x_2.npy')
    images_3 = np.load(FLAGS.data_dir + 'test_x_3.npy')
    images_4 = np.load(FLAGS.data_dir + 'test_x_4.npy')
    valid_images = [images_1, images_2, images_3, images_4]
    
    labels_1 = np.load(FLAGS.data_dir + 'test_y_1.npy')
    labels_2 = np.load(FLAGS.data_dir + 'test_y_2.npy')
    labels_3 = np.load(FLAGS.data_dir + 'test_y_3.npy')
    labels_4 = np.load(FLAGS.data_dir + 'test_y_4.npy')
    valid_labels = [labels_1, labels_2, labels_3, labels_4]

    train_num_examples = [train_images.shape[0], train_images.shape[1], train_images.shape[2], train_images.shape[3]]
    valid_num_examples = [valid_images.shape[0], valid_images.shape[1], valid_images.shape[2], valid_images.shape[3]] 

    # specify the network
    tf.reset_default_graph()
    with tf.name_scope('Input_Placeholder') as scope:
        x = tf.placeholder(shape=[None, 16641], name='input_placeholder', dtype=tf.float32)
        y = tf.placeholder(shape=[None, 7], name='output', dtype=tf.float32)

    with tf.name_scope('3_layer_cnn_model') as scope:
        output_layer = model.conv_block(x, [32,64,128], [3,3,3], tf.nn.relu, 0.5)
        logits = tf.identity(output_layer,name='output_hmm')

    # Calculate loss
    with tf.name_scope('Loss') as scope:
        loss_op = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
        reduce_mean = tf.reduce_mean(loss_op)

    # Training optimizer
    with tf.name_scope('Train') as scope:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001, name='Adam_Op')
        train_op = optimizer.minimize(loss=reduce_mean,
                             global_step=tf.train.get_global_step())

    global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer) # This is for allowing to keep track how far it is optimized so far.
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.summary.FileWriter("/work/cse496dl/cpack/Assignment_2/logs", tf.get_default_graph()).close()
        
        # run training
        min_validation_ce = float('INF')
        idx_min_validation_ce = 0
        
        for epoch in range(FLAGS.max_epoch_num):
            # Early stopping
            if epoch > 0 and min_validation_ce < avg_validation_ce_vals[-1] and epoch-idx_min_validation_ce>FLAGS.tol_epoch_num:
                print('No more loss reduction on validation set over ' + str(FLAGS.tol_epoch_num) + ' epochs.')
                print('Training process is terminated..')
                break
                
            print('Epoch: ' + str(epoch))

            # run gradient steps and report mean loss on train data
            ce_vals = []
            for i in range(train_num_examples // FLAGS.batch_size):
                batch_xs = train_images[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size, :]
                batch_ys = train_labels[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size, :]
                #if epoch%2 == 0:
                #    s, _ = sess.run([merged_summary,_], {x: batch_xs, y: batch_ys})
                #    writer.add_summary(s, epoch)
                _, train_ce = sess.run([train_op, reduce_mean], {x: batch_xs, y: batch_ys})
                ce_vals.append(train_ce)
                avg_train_ce = sum(ce_vals) / len(ce_vals)
            print('TRAIN CROSS ENTROPY: ' + str(avg_train_ce))
            
            # run validation set
            ce_vals = []
            avg_validation_ce_vals = []
            for i in range(validation_num_examples // FLAGS.batch_size):
                batch_xs = validation_images[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size, :]
                batch_ys = validation_labels[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size, :]
                validation_ce = sess.run(reduce_mean, {x: batch_xs, y: batch_ys})
                ce_vals.append(validation_ce)
            avg_validation_ce = sum(ce_vals) / len(ce_vals)
            avg_validation_ce_vals.append(avg_validation_ce)
            # update minimum validation_ce
            if avg_validation_ce < min_validation_ce:
                min_validation_ce = avg_validation_ce
                idx_min_validation_ce = epoch
                #path_prefix = saver.save(sess, os.path.join(FLAGS.save_dir, "fmnist_inference_chulwoo"), global_step=global_step_tensor)
                print('New model has been saved!')
            print('VALIDATION CROSS ENTROPY: ' + str(avg_validation_ce))

            # report mean test loss
        
            FLAGS.dropout = 0
            ce_vals = []
            conf_mxs = []
        #for i in range(test_num_examples // FLAGS.batch_size):
        #    batch_xs = test_images[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size, :]
        #    batch_xs = np.reshape(batch_xs,[-1,28,28,1])
        #    batch_ys = test_labels[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size, :]
        #    test_ce, conf_matrix = sess.run([reduce_mean, confusion_matrix_op], {x: batch_xs, y: batch_ys})
        #    ce_vals.append(test_ce)
        #    conf_mxs.append(conf_matrix)
        #avg_test_ce = sum(ce_vals) / len(ce_vals)
        #print('TEST CROSS ENTROPY: ' + str(avg_test_ce))
        #print('TEST CONFUSION MATRIX:')
        #print(str(sum(conf_mxs)))

if __name__ == '__main__':
    tf.app.run()
        
        
