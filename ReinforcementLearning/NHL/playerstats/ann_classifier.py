import pickle
from copy import deepcopy
from os import path

import numpy
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


class ANN_classifier():

    def __init__(self, nNodes=[10,20,2]):
        # Launch the builder
        nodes           =   deepcopy(nNodes)
        self.nInputs    =   nodes.pop(0)
        self.nOutputs   =   nodes.pop(-1)
        self.nNodes     =   nodes

    def ann_train_network(self, nIter, annI, annT, svname=None):

        # === FIRST: build the network
        # Architecture - 1 layer
        annX        =   tf.placeholder(tf.float32, [None, self.nInputs], name='Input_to_the_network-player_features')
        annY_       =   tf.placeholder(tf.float32, [None, 2], name='Ground_truth')
        annW1       =   tf.Variable(tf.truncated_normal([self.nInputs, self.nNodes[0]], stddev=0.1), name='weights_inp_hid')
        annB1       =   tf.Variable(tf.ones([1, self.nNodes[0]]) / 10, name='bias_inp_hid')
        Y1          =   tf.add(tf.nn.relu(tf.matmul(annX, annW1)), annB1, name='hid_output')
        annW2       =   tf.Variable(tf.truncated_normal([self.nNodes[0], self.nOutputs], stddev=0.1), name='weights_hid_out')
        annB2       =   tf.Variable(tf.ones([1, self.nOutputs]) / 10, name='bias_hid_out')
        annY        =   tf.add(tf.matmul(Y1, annW2), annB2, name='prediction')
        # Init variables
        init        =   tf.global_variables_initializer()
        # Optimization
        loss        =   tf.reduce_mean(tf.squared_difference(annY_, annY))
        train_step  =   tf.train.GradientDescentOptimizer(0.1).minimize(loss)
        # Compute accuracy
        is_correct  =   tf.equal(tf.argmax(annY, axis=1), tf.argmax(annY_, axis=1))
        accuracy    =   tf.reduce_mean(tf.cast(is_correct, tf.float32))


        self.trLoss, self.tsLoss, self.trAcc, self.tsAcc, nIter = [], [], [], [], 50
        # Initialize the model saver
        #builder         =   SavedModelBuilder(svname)
        saver           =   tf.train.Saver()
        for iIt in range(nIter):
            # --- TRAIN ANN
            # Init instance
            init    =   tf.global_variables_initializer()
            sess    =   tf.Session()
            sess.run(init)
            # Split train/test data
            train_X, test_X, train_Y, test_Y = train_test_split(annI, annT, test_size=0.25)
            # Loop on data splits
            fcnLoss =   []
            # Make exponential batch size increase
            batchSize, minSize, maxSize, nSteps =   [], 5, 20, 0
            while np.sum(batchSize) + maxSize < train_X.shape[0]:
                nSteps      +=  1
                batchSize   =   np.floor( ((np.exp(range(nSteps)) - 1) / (np.exp(nSteps) - 1)) ** .05 * (maxSize - minSize)) + minSize
            batchSize       =   np.append(batchSize, train_X.shape[0] - np.sum(batchSize)).astype(int)
            trL, tsL, trA, tsA  =   [], [], [], []
            for ib in batchSize:
                # Slice input and target
                trInput, train_X    =   train_X[:ib, :], train_X[ib:, :]
                trTarget, train_Y   =   train_Y[:ib, :], train_Y[ib:, :]
                # Pass them through
                dictDT  =   {annX: trInput, annY_: trTarget}
                sess.run(train_step, feed_dict=dictDT)
                trL.append(sess.run(loss, feed_dict=dictDT))
                trA.append(sess.run(accuracy, feed_dict=dictDT))
                # Assess accuracy
                dictTS  =   {annX: test_X, annY_: test_Y}
                tsL.append(sess.run(loss, feed_dict=dictTS))
                tsA.append(sess.run(accuracy, feed_dict=dictTS))
            # plt.figure(); plt.plot(fcnLoss)
            self.trLoss.append(trL)
            self.tsLoss.append(tsL)
            self.trAcc.append(trA)
            self.tsAcc.append(tsA)
            self.batchSize  =   batchSize
        # Save session
        if not svname is None:
            """
            builder.add_meta_graph_and_variables(self.sess, [tag_constants.SERVING])
            builder.save()
            """
            saver.save(sess, path.join(svname, 'MODEL_perceptron_1layer_10units_relu'))
            pickle.dump({'trLoss':self.trLoss, 'tsLoss':self.tsLoss, 'trAcc':self.trAcc, 'tsAcc':self.tsAcc, 'batchSize':self.batchSize}, \
                        open(path.join(svname, 'addedVariables.p'), 'wb') )

    def ann_display_accuracy(self):
        # Make figure
        Fig     =   plt.figure()
        # Axes1: loss
        Ax1     =   Fig.add_subplot(121)
        Ax1.fill_between(np.cumsum(self.batchSize), np.mean(self.trLoss, axis=0) - np.std(self.trLoss, axis=0), np.mean(self.trLoss, axis=0) + np.std(self.trLoss, axis=0), facecolors='b', interpolate=True, alpha=0.4)
        Ax1.fill_between(np.cumsum(self.batchSize), np.mean(self.tsLoss, axis=0) - np.std(self.tsLoss, axis=0), np.mean(self.tsLoss, axis=0) + np.std(self.tsLoss, axis=0), facecolors='r', interpolate=True, alpha=0.4)
        Ax1.plot(np.cumsum(self.batchSize), np.mean(self.trLoss, axis=0), 'b')
        Ax1.plot(np.cumsum(self.batchSize), np.mean(self.tsLoss, axis=0), 'r')
        Ax1.set_xlabel('Number of training examples')
        Ax1.set_ylabel('Quadratic error')
        Ax1.set_xlim([min(self.batchSize), np.sum(self.batchSize)])
        Ax1.set_ylim([0.04, 0.16])
        # Axes2: accuracy
        Ax2     =   Fig.add_subplot(122)
        Ax2.fill_between(np.cumsum(self.batchSize), np.mean(self.trAcc, axis=0) - np.std(self.trAcc, axis=0), np.mean(self.trAcc, axis=0) + np.std(self.trAcc, axis=0), facecolors='b', interpolate=True, alpha=0.4)
        Ax2.fill_between(np.cumsum(self.batchSize), np.mean(self.tsAcc, axis=0) - np.std(self.tsAcc, axis=0), np.mean(self.tsAcc, axis=0) + np.std(self.tsAcc, axis=0), facecolors='r', interpolate=True, alpha=0.4)
        Ax2.plot(np.cumsum(self.batchSize), np.mean(self.trAcc, axis=0), 'b')
        Ax2.plot(np.cumsum(self.batchSize), np.mean(self.tsAcc, axis=0), 'r')
        Ax2.set_xlabel('Number of training examples')
        Ax2.set_ylabel('Classification accuracy')
        Ax2.set_xlim([min(self.batchSize), np.sum(self.batchSize)])
        Ax2.set_ylim([0.4, 1])

    def ann_forward_pass(self, repoModel, input_data):
        # Restore model
        sess, annX, annY = self.ann_reload_model(repoModel)
        # Restore additional variables
        VAR = pickle.load(open(path.join(repoModel, 'addedVariables.p'), 'rb'))
        self.trLoss = VAR['trLoss']
        self.tsLoss = VAR['tsLoss']
        self.trAcc = VAR['trAcc']
        self.tsAcc = VAR['tsAcc']
        self.batchSize = VAR['batchSize']
        return sess.run(annY, feed_dict={annX: input_data})

    def ann_reload_model(self, repoModel):
        # Reload the graph and variables
        sess = tf.Session()
        saver = tf.train.import_meta_graph(path.join(repoModel, path.basename(repoModel) + '.meta'))
        saver.restore(sess, tf.train.latest_checkpoint(path.join(repoModel, './')))
        # Link TF variables to the classifier class
        graph = sess.graph
        annX = graph.get_tensor_by_name('Input_to_the_network-player_features:0')
        """self.annY_  =   graph.get_tensor_by_name('Ground_truth:0')
        self.annW1  =   graph.get_tensor_by_name('weights_inp_hid:0')
        self.annB1  =   graph.get_tensor_by_name('bias_inp_hid:0')
        self.Y1     =   graph.get_operation_by_name('hid_output')
        self.annW2  =   graph.get_tensor_by_name('weights_hid_out:0')
        self.annB2  =   graph.get_tensor_by_name('bias_hid_out:0')"""
        annY = graph.get_tensor_by_name('prediction:0')
        return sess, annX, annY