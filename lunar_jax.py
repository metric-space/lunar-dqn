import gymnasium as gym
import numpy as np
import random as r

from matplotlib import pyplot
from collections import deque
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import debug
from jax.scipy.special import logsumexp


env = gym.make("LunarLander-v2")


# ----------------------------------------------------------------------
# 
#          Constants
#
# ----------------------------------------------------------------------


STEP_SIZE            = 5e-4
ARCHITECTURE         = [8, 64, 64, env.action_space.n]
N_EPISODES           = 1000 # 10000

START_EPSILON        = 1
EPISODE_DECAY_LENGTH = 400
LINEAR_DECAY_STEP    = START_EPSILON / EPISODE_DECAY_LENGTH
FINAL_EPSILON        = 0.1
BATCH_SIZE           = 64

REPLAY_MEMORY_LENGTH = 100000

epsilon_decay_func   = lambda epsilon: max(FINAL_EPSILON, epsilon - LINEAR_DECAY_STEP)

GRAPH_FILE           = 'dqn.png'

RAND                 = random.PRNGKey(146543)


# ----------------------------------------------------------------------
# 
#          Neural Network stuff (almost the same code from jax tutorial)
#
# ----------------------------------------------------------------------


def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


def relu(x):
  return jnp.maximum(0, x)


def predict(params, input):
    activations = input
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return logits


batched_predict = vmap(predict, in_axes=(None, 0))


def loss(params, images, targets, actions):
  preds = jnp.take_along_axis(batched_predict(params, images), actions, axis=1)
  preds = jnp.squeeze(preds, axis=1)
  return jnp.mean((preds - targets)**2)


@jit
def update(params, x, y, actions, step_size):
  grads = grad(loss)(params, x, y, actions)
  return [(w - step_size * dw, b - step_size * db) for (w, b), (dw, db) in zip(params, grads)]


# ----------------------------------------------------------------------
# 
#          Constants
#
# ----------------------------------------------------------------------


class NeuralNet(object):
    def __init__(self, architecture, key):
        self.params = init_network_params(architecture, key)

    
# split from somewhere

class Buffer:
    def __init__(self,length):
        self.buffer = deque(maxlen=length)

    def append(self,x):
        self.buffer.append(x)

    def sample(self,k):
        return r.choices(self.buffer, k=k)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self,x):
        return self.buffer[x]

def gen_state(env, key, model_local, model_target, buffer_length):
    return {
            'qmodel_local'  : model_local,
            'qmodel_target' : model_target,
            'replay_buffer' : Buffer(buffer_length),
            'env'           : env,
            'state'         : None,
            'tau'           : jnp.float32(0.001),
            'key'           : key
        }


def initialize(state, n_episodes, start_epsilon):
    state['env'] = gym.wrappers.RecordEpisodeStatistics(state['env'], deque_size=n_episodes)
    state['episode_counter'] = n_episodes
    state['epsilon'] = start_epsilon


def episode_run(state, batch_size, step_size, additional_func):
    done = False
    observation, _ = state['env'].reset()
    state['state'] = observation

    while not done:
        action = None

        new_keys = random.split(state['key'],2)
        state['key'] = new_keys[0]

        if random.uniform(new_keys[1]) < state['epsilon']:
            action = state['env'].action_space.sample()
        else:
            action = predict(Q.params, state['state'])
            action = jnp.argmax(action).item()
    
        observation2, reward, terminated, truncated, info = state['env'].step(action)

        done = terminated or truncated
    
        state['replay_buffer'].append((state['state'],action,reward, observation2, done))
    
        state['state'] = observation2
    
        if len(state['replay_buffer']) < batch_size:
            continue
    
        rewards     = []
        states      = []
        next_states = []
        dones       = []
        actions     = []
        
        samples = state['replay_buffer'].sample(batch_size)
    
        for sample in samples:
            (s_t0, a, r, s_t1 , term) = sample
            dones.append(term)
            rewards.append(r)
            states.append(s_t0)
            next_states.append(s_t1)
            actions.append(a)

        rewards     = jnp.array(rewards)
        states      = jnp.array(states)
        actions     = jnp.vstack(jnp.array(actions))
        dones       = jnp.array(dones)
        next_states = jnp.array(next_states)


        predicted = jnp.max(batched_predict(state['qmodel_target'].params ,jnp.array(next_states)),axis=1)
        y_ = rewards + 0.99 * predicted*(1 - dones)

        state['qmodel_local'].params = update(state['qmodel_local'].params, states, y_, actions, step_size)

        l = state['qmodel_local'].params
        t = state['qmodel_target'].params

        state['qmodel_target'].params = [
                (
                    state['tau']*l[i][0] + (1-state['tau'])*t[i][0],
                    state['tau']*l[i][1] + (1-state['tau'])*t[i][1]
                )  for i in range(len(l))]


    additional_func(state['env'])


def plot_func(env):
    l = env.return_queue
    pyplot.plot(range(len(l)),l)
    pyplot.savefig(GRAPH_FILE)


keys = random.split(RAND, 3)

Q = NeuralNet(ARCHITECTURE, keys[1])
T = NeuralNet(ARCHITECTURE, keys[2])

state = gen_state(env,keys[0], Q,T, REPLAY_MEMORY_LENGTH)
initialize(state, N_EPISODES, START_EPSILON)

while state['episode_counter'] >= 0:
    episode_run(state, BATCH_SIZE , STEP_SIZE, plot_func)
    state['epsilon'] = epsilon_decay_func(state['epsilon'])
    state['episode_counter'] -= 1
