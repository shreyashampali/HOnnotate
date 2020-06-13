from ghope.common import *
from ghope.utils import *
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

class Optimizer():
    def __init__(self, loss, varList, type='Adam', learning_rate=0.01, initVals = None):
        self.globalStep =  tf.Variable(0, name='global_step', trainable=False)
        self.type = type
        self.loss = loss
        if type == 'Adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.999)
        elif type=='Adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=1.0)
        elif type == 'Adagrad':
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
        elif type == 'Momentum':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        elif type == 'BFGS':
            self.varsTensor = tf.concat(varList, axis=0)
            self.optOp = tfp.optimizer.bfgs_minimize(
                self.funcBFGS, initial_position=initVals, tolerance=1e-8)
        else:
            raise Exception(' Optimizer %s not Implemented!!' % (type))

        self.optOp = self.optimizer.minimize(loss, var_list=varList)
        self.grads = self.optimizer.compute_gradients(loss, varList)

        self.numVars = 0
        for var in varList:
            self.numVars += int(np.prod(var.shape))

        if type == 'Adam':
            self.resetInternalTFVars()

    def funcBFGS(self, varsTensor):
        grads = tf.gradients(self.loss, varsTensor)
        return self.loss, grads

    def resetInternalTFVars(self):
        varList = []
        if self.type=='Adam':
            vars = self.optimizer.variables()
            for var in vars:
                if 'beta' in var.name:
                    varList.append(var)
        self.optIntResetOp = tf.variables_initializer(varList)



    @timeit
    def runOptimization(self, session, steps, feedDict=None, logLossFunc=False, lossPlotName=''):
        if self.type != 'BFGS':
            lossCurve = np.zeros((steps,), dtype=np.float32)
            gradCurve = np.zeros((self.numVars, steps), dtype=np.float32)
            for i in range(steps):
                if logLossFunc:
                    lt = session.run(self.loss, feed_dict=feedDict)
                    lossCurve[i] = lt
                    ind = 0
                    for it in self.grads:
                        grad = session.run(it[0], feed_dict=feedDict)
                        grad = grad.reshape(-1)
                        for j in range(len(grad)):
                            gradCurve[ind, i] = grad[j]
                            ind = ind + 1
                session.run(self.optOp, feed_dict=feedDict)
                if self.type == 'Adam':
                    session.run(self.optIntResetOp)

            if logLossFunc:
                plt.ioff()
                fig, ax = plt.subplots(1, 2)
                ax[0].plot(np.arange(0, steps), lossCurve)
                for i in range(self.numVars):
                    ax[1].plot(np.arange(0,steps), gradCurve[i])

                plt.savefig(lossPlotName)
                plt.close(fig)
                plt.ion()
        elif self.type == 'BFGS':
            results = session.run(self.optOp)
            assert(results.converged)
            print("Function evaluations: %d" % results.num_objective_evaluations)
            a = 10

