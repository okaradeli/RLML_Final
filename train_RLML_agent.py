#!/usr/bin/env python

import time
import IHML.globals as glb

import agent_RLML as agent
from env_RLML import IHML_Env
from IHML.incremental_metalearner import IncrementalMetalearner
import IHML.incremental_metalearner as ihml_base
# agent_class defines the following classes:
# - "neural_network"
# - "memory"
# - "agent_base" with derived classes
#   - "dqn" for (double) deep Q-learning
#   - "actor_critic" for the actor-critic learning algorithm

#Global loger
import logging
logger = logging.getLogger()

class Train_IHML():

    overwrite=True
    dqn=True
    ddqn=False
    env=None

    def train_ihml(self,XTrain,XTest,YTrain,YTest):

        # Create environment
        self.env = IHML_Env(XTrain,YTrain)

        # Obtain dimensions of action and observation space
        N_actions = len(self.env.action_space())
        observation, info = self.env.reset()
        N_state = len(observation)
        print('dimension of state space =',N_state)
        print('number of actions =',N_actions)

        # Set parameters
        # NOTE: Only the first two parameters (N_state and N_actions) are mandatory,
        # the reminaing parameters are optional.
        # For demonstration, we here set all algorithm-independent optional parameters
        # to their default. Because for all the extra parameters below we use their
        # default values, using
        #      parameters = {'N_state':N_state, 'N_actions':N_actions}
        # instead of the dictionary below will yield the same results.
        #
        parameters = {
            # Mandatory parameters
            'N_state':N_state,
            'N_actions':N_actions,
            #
            # All the following parameters are optional, and we set them to
            # their default values here:
            #
            'discount_factor':0.99, # discount factor for Bellman equation
            #
            'N_memory':20000, # number of past transitions stored in memory
                                # for experience replay
            #
            # Optimizer parameters
            'training_stride':1, # number of simulation timesteps between ##10
                # optimization (learning) steps
            'batch_size':4,##'batch_size':32, # mini-batch size for optimizer
            'saving_stride':100, # every saving_stride episodes, the
                # current status of the training is saved to disk
            #
            # Parameters for stopping criterion for training
            'n_episodes_max':glb.MAX_EPOCH, # maximal number of episodes until the #100
                # training is stopped (if stopping criterion is not met before)
            'n_solving_episodes':glb.MAX_EPOCH, # the last N_solving episodes need to
                # fulfill both:
            # i) minimal return over last N_solving_episodes:
            'solving_threshold_min':200.,
            # ii) mean return over last N_solving_episodes:
            'solving_threshold_mean':230.,
                }

        # Instantiate agent class
        logger.info("Started training...")
        my_agent = agent.dqn(parameters,self.env.action_space_all)
        self.env.agent=my_agent


        # Train agent on environment
        start_time = time.time()
        training_results = my_agent.train(environment=self.env, model_filename=glb.MODEL_DATA_FILE_NAME)
        logger.info("Training complete.")
        execution_time = (time.time() - start_time)
        logger.info('Execution time in seconds: ' + str(execution_time))
        logger.info("Training results"+str(training_results))



if __name__ == '__main__':
    logger.info("Train_IHML started standalone bro")
    #Dataset load
    # Base algorithm
    ihml = IncrementalMetalearner()
    XTrain,XTest,YTrain,YTest = ihml.load_dataset()
    train_IHML= Train_IHML( )
    train_IHML.train_ihml(XTrain,XTest,YTrain,YTest)