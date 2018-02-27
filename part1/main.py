#author: Chulwoo Pack
import model
import util
import tensorflow as tf
import numpy as np
import math

#code to set flags by Paul Quint
flags = tf.app.flags
flags.DEFINE_string('data_dir', '/work/cse496dl/shared/homework/02/SAVEE-British/', 'directory where dataset is located')
flags.DEFINE_string('save_dir', '/work/cse496dl/cpack/Assignment_2/transfer/set1_noReg_50/savee_homework_2', 'directory where model graph and weights are saved')
flags.DEFINE_string('load_Dir', '/work/cse496dl/cpack/Assignment_2/set1_noReg_50/', 'directory where model graph and weights are saved')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('max_epoch_num', 50, '')
FLAGS = flags.FLAGS

def main(argv):
    #load data
    test_data_1 = np.load(FLAGS.data_dir + 'test_x_1.npy')
    test_data_2 = np.load(FLAGS.data_dir + 'test_x_2.npy')
    test_data_3 = np.load(FLAGS.data_dir + 'test_x_3.npy')
    test_data_4 = np.load(FLAGS.data_dir + 'test_x_4.npy')
    test_data = [test_data_1, test_data_2, test_data_3, test_data_4]

    test_labels_1 = np.load(FLAGS.data_dir + 'test_y_1.npy')
    test_labels_2 = np.load(FLAGS.data_dir + 'test_y_2.npy')
    test_labels_3 = np.load(FLAGS.data_dir + 'test_y_3.npy')
    test_labels_4 = np.load(FLAGS.data_dir + 'test_y_4.npy')
    test_labels = [test_labels_1, test_labels_2, test_labels_3, test_labels_4]

  
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
    test_count = [test_data[0].shape[0], test_data[1].shape[0], test_data[2].shape[0], test_data[3].shape[0]]
    train_count = [train_data[0].shape[0], train_data[1].shape[0], train_data[2].shape[0], train_data[3].shape[0]]

    # Load model
    with tf.Session() as session:
        saver = tf.train.import_meta_graph(FLAGS.load_Dir + 'emodb_homework_2-0-0.meta')
        saver.restore(session, FLAGS.load_Dir + 'emodb_homework_2-0-0')
    
        graph = session.graph
        input_placeholder = graph.get_tensor_by_name('input_placeholder:0')
        labels = tf.placeholder(shape=[None, 7], name='output2', dtype=tf.float32)
        conv_out = graph.get_tensor_by_name('Conv_Pool_3/pool_3/MaxPool:0')

        #  TRANSFER LEARNING
        conv_out_no_gradient = tf.reshape(tf.stop_gradient(conv_out), [-1, 16*16*128])
        #our_dropout = tf.layers.dropout(inputs=conv_out_no_gradient, rate=0.5, name="our_dropout")
        our_dense_layer = tf.layers.dense(conv_out_no_gradient, units=7, name="our_dense_layer")
        
        # TRAINING SETTING
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=our_dense_layer)
        confusion_matrix_op = tf.confusion_matrix(tf.argmax(labels,axis=1), tf.argmax(our_dense_layer, axis=1), num_classes=7)
        global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
        saver = tf.train.Saver()
        sum_cross_entropy = tf.reduce_mean(cross_entropy)
            
        optimizer = tf.train.AdamOptimizer(learning_rate=0.005, name="Adam_Op")
        train_op = optimizer.minimize(loss=cross_entropy, global_step=global_step_tensor)
        
        best_test_conf_mxs = []
        best_epoch = [0, 0, 0, 0]
        best_test_ce = [10, 10, 10, 10]
        best_train_ce = [0, 0, 0, 0]
        best_classification_rate = [0, 0, 0, 0]
        epochs_since_best = [0, 0, 0, 0]

        # TRAINING START
        print("################### TRANSFER LEARNING ###################")
        
        for k in range(0,4):
            session.run(tf.global_variables_initializer())
            # initialize new variables
            optimizer_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "optimizer")
            #dropout_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "our_dropout")
            dense_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "our_dense_layer")
            #session.run(tf.variables_initializer(optimizer_vars + dropout_vars + dense_vars, name='init'))
            session.run(tf.variables_initializer(optimizer_vars + dense_vars, name='init'))
            
            batch_size = FLAGS.batch_size
            print("\n !!!!! NEW K (" + str(k) + ") !!!!!\n")
            for epoch in range(FLAGS.max_epoch_num):
                print("################### EPOCH " + str(epoch) + " #####################")
                print("##################################################\n")
        
                # run gradient steps and report mean loss on train data
                ce_vals = []
                for i in range(train_count[k] // batch_size):
                    batch_data = train_data[k][i*batch_size:(i+1)*batch_size, :]
                    batch_labels = train_labels[k][i*batch_size:(i+1)*batch_size]
                    _, train_ce = session.run([train_op, sum_cross_entropy], {input_placeholder: batch_data, labels: batch_labels})
                    ce_vals.append(train_ce)
                avg_train_ce = sum(ce_vals) / len(ce_vals)
                best_train_ce[k] = avg_train_ce
                print('TRAIN CROSS ENTROPY: ' + str(avg_train_ce))
                
            print("\n################# TESTING... ###################\n")
            ce_vals = []
            conf_mxs = []
            for i in range(test_count[k] // batch_size):
                batch_data = test_data[k][i*batch_size:(i+1)*batch_size, :]
                batch_labels = test_labels[k][i*batch_size:(i+1)*batch_size]
                test_ce, conf_matrix = session.run([sum_cross_entropy, confusion_matrix_op], {input_placeholder: batch_data, labels: batch_labels})
                ce_vals.append(test_ce)
                conf_mxs.append(conf_matrix)
            avg_test_ce = sum(ce_vals) / len(ce_vals)
            classification_rate = util.classification_rate(sum(conf_mxs),7)
            print('TEST CROSS ENTROPY: ' + str(avg_test_ce))
            print('TEST CONFUSION MATRIX:')
            print(str(sum(conf_mxs)))
            print('TEST CLASSIFICATION RATE:' + str(classification_rate))
            best_test_conf_mxs.append(sum(conf_mxs))
            best_test_ce[k] = avg_test_ce
            best_classification_rate[k] = classification_rate

        print("\n############# END OF K-FOLD ##################") 
        print('Confusion Matrix: ')
        print(str(sum(best_test_conf_mxs)))
        print('Avg Test CE: ' + str(np.average(best_test_ce)))
        print('Avg Train CE: ' + str(np.average(best_train_ce)))
        print('Avg Classification Rate: ' + str(np.average(best_classification_rate)))
        print('Generating model now...')
        session.run(tf.global_variables_initializer())
        
        for j in range(0,4):
            for epoch in range(FLAGS.max_epoch_num):
                for i in range(train_count[j] // batch_size):
                    batch_data = train_data[j][i*batch_size:(i+1)*batch_size, :]
                    batch_labels = train_labels[j][i*batch_size:(i+1)*batch_size]
                    _, train_ce = session.run([train_op, sum_cross_entropy], {input_placeholder: batch_data, labels: batch_labels})

        #saver.save(session, FLAGS.save_dir + 'savee_homework_2')
        saver.save(session, FLAGS.save_dir)
        print('Model is generated and saved')

if __name__ == "__main__":
    tf.app.run()