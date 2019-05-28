from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype

from keras.layers import Input, Dense, Activation, BatchNormalization, Dropout, InputLayer
from keras.models import Model, Sequential
from keras.layers import LeakyReLU

class PushRecoveryPolicy(object):
    recurrent = False
     

    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name
            
    def _init(self, ob_space, ac_space, n_prev_obs, hidden_units, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        obz = ob

        # Critic
        criticNN =  Sequential()
        criticNN.add(InputLayer(input_tensor = obz)
        criticNN.add(Reshape((n_prev_obs, ob_space.shape[1] // n_prev_obs)))
        criticNN.add(LSTM(hidden_units))

        self.vpred = U.dense(criticNN.output, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:,0]

        # Actor
        actorNN =  Sequential()
        actorNN.add(InputLayer(input_tensor = obz)
        actorNN.add(Reshape((n_prev_obs, ob_space.shape[1] // n_prev_obs)))
        actorNN.add(LSTM(hidden_units))


        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            mean = actorNN.output            
            logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer(), trainable= False)
            pdparam = U.concatenate([mean, mean * 0.0 + logstd], axis=1)
        else:
            pdparam = U.dense(model.output, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)
        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob], [ac, self.vpred])
