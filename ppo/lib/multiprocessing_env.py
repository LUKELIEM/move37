#This code is from openai baseline
#https://github.com/openai/baselines/tree/master/baselines/common/vec_env

# OpenAI's Vectorized Environments are a method for stacking multiple independent environments 
# into a single environment. Instead of training an RL agent on 1 environment per step, it allows 
# us to train it on n environments per step.

# 7-28-2019 Added documentation while learning the code

import numpy as np
from multiprocessing import Process, Pipe

# In multiprocessing, processes are spawned by creating a Process object and then calling start().

# multiprocessing supports two types of communication channel between processes:
#   Queue() - a near clone of queue.Queue
#   Pipe() - returns a pair of connection objects connected by a duplex (two-way) pipe.
#     Each connection object has send() and recv() methods (among others).


def worker(remote, parent_remote, env_fn_wrapper):
    
    parent_remote.close()
    env = env_fn_wrapper.x()

    while True:

        cmd, data = remote.recv()  # get cmd and data

        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            # send the environ's observation and actions
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError

class VecEnv(object):
    """
    An abstract asynchronous, vectorized environment.
    """
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a tuple of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    
class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

        
class SubprocVecEnv(VecEnv):

    # Implementing multiprocessing in PyTorch is explained in:
    # https://pytorch.org/docs/stable/notes/multiprocessing.html

    # Creates a multiprocess vectorized wrapper for multiple environments, distributing each 
    # environment to its own process, allowing significant speed up when the environment is 
    # computationally complex.

    # Note that with Pipe(), blocking pt-2-pt communication (send() and recv()) is used 

    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """

        # Flags for multiprocessing
        self.waiting = False
        self.closed = False
        
        nenvs = len(env_fns)
        self.nenvs = nenvs
        
        # in these environs, remotes and workremotes are connected by two-way pipes
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])

        # Set up processes based on the environs, remotes and workremotes. Process() will call
        # worker function with parameters:
        #   remote = work_remote
        #   parent_remote = remote
        #   env_fn_wrapper = CloudpickleWrapper(env_fn)
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        
        for p in self.ps:
            # Note: The process’s daemon flag must be set before start() is called. The default is 
            # for processes to not be daemons
            p.daemon = True # Set process to daemon, so if the main process crashes, it will not cause things to hang
            p.start()
        
        # close the connection on the work_remote end
        for remote in self.work_remotes:
            remote.close()

        # Get remote work 0 to send back obs and action
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()

        # Initialize VecEnv object ???
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        # Tell all the environments to start taking a step with the given actions. 
        # Call step_wait() to get the results of the step.

        # Send 'step' cmd to remote workers
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))   # Note that this is blocking pt-2-pt communication

        self.waiting = True   # Set waiting flag to True

    def step_wait(self):
        # Wait for the step taken with step_async().

        # Note that blocking pt-2-pt communication is used
        results = [remote.recv() for remote in self.remotes]

        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
            self.closed = True
            
    def __len__(self):
        return self.nenvs
