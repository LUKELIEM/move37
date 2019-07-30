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

# OpenAI implements vectorized environs by pairing a parent to a worker using Pipe().


def worker(remote, parent_remote, env_fn_wrapper):
    # In this function, the worker receives a cmd from its parent on the other end of Pipe and
    # acts upon the cmd.

    parent_remote.close()  # close the connection on the parent's end
    env = env_fn_wrapper.x() 

    while True:

        cmd, data = remote.recv()  # worker receives cmd and data from parent

        if cmd == 'step':
            # worker steps thru environ
            ob, reward, done, info = env.step(data)
            
            if done:
                ob = env.reset()  # reset environ if done

            # worker sends ob, reward, done, info to parent
            remote.send((ob, reward, done, info))  

        elif cmd == 'reset':
            # worker resets environ
            ob = env.reset() 

            # worker sends ob back to parent
            remote.send(ob) 

        elif cmd == 'reset_task':
            # reset environ task
            ob = env.reset_task()

            # worker sends ob back to parent
            remote.send(ob)

        elif cmd == 'close':
            # worker closes connection, then break to exit
            remote.close()
            break

        elif cmd == 'get_spaces':
            # worker sends the environ's obs_space and action_space to parent
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
        
        # For the multiple environs, set up pipes connecting parents (remotes) and workers (work_remotes)
        nenvs = len(env_fns)
        self.nenvs = nenvs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])

        # Set up a list of processes, whereby Process() call worker function with parameters:
        #   remote = work_remotes (worker)
        #   parent_remote = remotes (parent)
        #   env_fn_wrapper = CloudpickleWrapper(env_fn)

        # Naming is horrible!!! Going forward, we will use parent and worker to denote the 2 ends of the pipe:
        #   parent - self.remotes
        #   worker - self.work_remotes
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        
        for p in self.ps:
            # Note: The process’s daemon flag must be set before start() is called. The default is 
            # for processes to not be daemons
            p.daemon = True # Set process to daemon, so if the main process crashes, it will not cause things to hang
            p.start()
        
        # workers close their connections
        for remote in self.work_remotes:
            remote.close()

        # parent send 'get_spaces' to worker, and recv from it obs_space and action_space
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()

        # Initialize VecEnv object ???
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    """
    In the methods below, the parents send cmds to their corresponding workers on the other end of the pipes.

    For each cmd, there is a corresponding action in worker().
    """

    def step_async(self, actions):
        # Tell all the environments to start taking a step with the given actions. 
        # Call step_wait() to get the results of the step.

        # Parents send 'step' cmd to workers
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

        self.waiting = True   # Set waiting flag to True

    def step_wait(self):
        # Wait for the step taken with step_async().

        # Parents wait for results from workers
        results = [remote.recv() for remote in self.remotes]

        self.waiting = False   # Set waiting flag to False

        # results consist of obs, rewards, dones and infos from workers' environs
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        # Parents send 'reset' cmds to works
        for remote in self.remotes:
            remote.send(('reset', None))
        # Parents wait for obs to come back from workers
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        # Parents send 'reset_task' cmds to works
        for remote in self.remotes:
            remote.send(('reset_task', None))
        # Parents wait for obs to come back from workers
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):

        # If closed flag is True, return
        if self.closed:
            return    

        # Otherwise
        if self.waiting:   # if waiting flag is still True
            # Parents wait to recv results from workers
            for remote in self.remotes:            
                remote.recv()

        # Parents send 'close' cmds to workers
        for remote in self.remotes:
            remote.send(('close', None))  

        # Terminate all processes, then set closed flag to True
        for p in self.ps:   
            p.join()
            self.closed = True
            
    def __len__(self):
        return self.nenvs
