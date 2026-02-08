#!/usr/bin/env python

import argparse
import torch
import numpy as np
import itertools 
import h5py 
import agent_RLML as agent


import IHML.globals as glb
from env_RLML import IHML_Env
from IHML.incremental_metalearner import IncrementalMetalearner
import IHML.incremental_metalearner as ihml_base

#Global loger
import logging
logger = logging.getLogger()


class Run_IHML():

    verbose=True



    def run_and_save_simulations(self,env,input_filename,n_episodes):
        #
        # load trained model
        input_dictionary = torch.load(open(input_filename,'rb'))
        dict_keys = np.array(list(input_dictionary.keys())).astype(int)
        max_index = np.max(dict_keys)
        input_dictionary = input_dictionary[max_index] # During training we
        # periodically store the state of the neural networks. We now use
        # the latest state (i.e. the one with the largest episode number), as
        # for any succesful training this is the state that passed the stopping
        # criterion
        #

        #1 episode run is sufficient for run ( testing )
        n_episodes=1

        # instantiate environment
        env.reset()

        # instantiate agent
        parameters = input_dictionary['parameters']

        #

        #
        durations = []
        returns = []
        last_action_value=0

        outputs={} ##The outputs of the experiment
        status_string = ("Run {0} of {1} completed with return {2:<5.1f}. Mean "
                "return over all episodes so far = {3:<6.1f}            ")
        # run simulations
        for i in range(n_episodes):
            logger.info("Starting EPISODE:"+str(i))
            # reset environment, duration, and reward
            state, info = env.reset()
            # Instantiate agent class
            my_agent = agent.dqn(parameters, env.action_space_all)
            my_agent.load_state(input_dictionary, train_mode=False)

            episode_return = 0.
            consecutive_negatives = 0 ##if adding a new feature/learner has negative value increment this
            previous_step_reward=0
            MAX_CONSEQUTIVE_NEGATIVES=50
            #
            # Variable to keep iterations going (i.e. no max_iteration, no out of actions...)
            done = False

            for n in itertools.count():


                #
                action_vector,action_new,no_more_action,action_value = my_agent.act(state)

                # Check reward is not improving
                ##if previous_step_reward > step_reward :
                if action_value < 0 :
                    consecutive_negatives += 1
                    if consecutive_negatives >=MAX_CONSEQUTIVE_NEGATIVES:
                        logger.info(str(MAX_CONSEQUTIVE_NEGATIVES)+ " consequtive negative values , we shall not add more features/learners")
                else:
                    consecutive_negatives = 0  # Reset counter if action_value is non-negative




                #Max number of iterations on an episode exceeded end iterations
                if n > glb.MAX_ITERATION:
                    logger.info("MAX ITERATION REACHED, ending episode:")
                    done=True

                # Unless there is no more action to melt, continue stepping
                if not done and not no_more_action:
                    state, step_reward, terminated, truncated, info = env.step(action_vector, action_new, i)
                    done = no_more_action or terminated or truncated or consecutive_negatives>=MAX_CONSEQUTIVE_NEGATIVES

                else:
                    done = True

                episode_return += step_reward
                #
                if done:
                    #
                    durations.append(n+1)
                    returns.append(step_reward)
                    last_action_value = action_value
                    #
                    if self.verbose:
                        if i < n_episodes-1:
                            end ='\r'
                        else:
                            end = '\n'
                        logger.info("Run:" + str(i + 1) + " of " + str(n_episodes) + " completed." + " Mean return over all episodes so far:" + str(np.mean(np.array(returns))))
                    logger.info("Final Reward:"+str(action_value)+" features:"+str(action_vector))
                    my_agent.log_action_state_reward(glb.EXPERIMENT_ID, 0, 0, 0, state,0, step_reward)
                    break
        #
        dictionary = {'returns':np.array(returns),
                    'durations':np.array(durations),
                    'input_file':input_filename,
                    'N':n_episodes}
        logger.info("Run complete, final iteration reward:"+str(step_reward))##str(np.max(returns)))

        #with h5py.File(output_filename, 'w') as hf:
        #    for key, value in dictionary.items():
        #        hf.create_dataset(str(key),
        #            data=value)
        outputs["IHML_RL_Duration"]=np.mean(durations)
        outputs["IHML_RL_Score"] = np.max(action_value)
        outputs["EXPERIMENT_ID"] = glb.EXPERIMENT_ID
        return outputs
    







if __name__ == '__main__':
    # Dataset load
    # Base algorithm
    ihml = IncrementalMetalearner()
    XTrain, XTest, YTrain, YTest = ihml.load_dataset()
    env = IHML_Env(XTest, YTest)

    logger.info("Run_IHML started standalone bro")
    run_ihml=Run_IHML()
    N = 3
    run_ihml.run_and_save_simulations(env=env, input_filename=glb.MODEL_DATA_FILE_NAME, n_episodes=N)
