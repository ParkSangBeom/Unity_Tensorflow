import tensorflow as tf
import numpy as np
import threading
import socket
import json
from random import *
import Saver as sv

PROJECT_NAME = "TensorFlow_Base"

HOST = "127.0.0.1"
PORT = 9051
BUFF_SIZE = 10000000

DISCOUNT = 0.99
EPSILON = 1e-8

EPISODE = 0
EPISODE_INTERVAL = 100

HEIGHT_SIZE = 10
WIDTH_SIZE = 10
DEPTH_SIZE = 1
INPUT_SIZE = HEIGHT_SIZE * WIDTH_SIZE * DEPTH_SIZE
OUTPUT_SIZE = 2

OBS_LIST = []
ACT_LIST = []
REW_LIST = []

ACTION = 0
STATE = "state_wait"

class ActorCriticNetwork:
    _critic_learning_rate = 0.001
    _actor_learning_rate = 0.001
    _hidden_size = 100

    def __init__(self, sess, height_size, width_size, depth_size,  output_size, name):
        self._sess = sess
        self._height_size = height_size
        self._width_size = width_size
        self._depth_size = depth_size
        self._output_size = output_size
        self._name = name

        self._BulidBaseNetwork()
        self._BulidCriticNetwork()
        self._BulidActorNetwork()

        self._sess.run(tf.global_variables_initializer())

        self._saver = sv.Saver(name, sess)
        self._SetSaver()

        print("TF준비 완료!")

    def _BulidBaseNetwork(self):
        #self._obs = tf.placeholder(tf.float32, shape=[None, self._input_size], name="obs_input")
        self._obs = tf.placeholder(dtype = tf.float32, shape = [None, self._height_size, self._width_size, self._depth_size], name="obs_input")
        self._action = tf.placeholder(tf.float32, shape=[None, self._output_size], name="action_input")
        self._reward = tf.placeholder(tf.float32, shape=[None, 1], name="reward_input")
        self._advantage_value = tf.placeholder(tf.float32, shape=[None, 1], name="advantage_input")

        with tf.name_scope("Conv1"):
            F1 = tf.Variable(tf.random_normal([3, 3, 1, 8], stddev=0.01))
            L1 = tf.nn.conv2d(self._obs, F1, strides=[1, 1, 1, 1], padding='VALID')
            L1 = tf.nn.relu(L1)
            L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        with tf.name_scope("Conv2"):
            F2 = tf.Variable(tf.random_normal([3, 3, 8, 16], stddev=0.01))
            L2 = tf.nn.conv2d(L1, F2, strides=[1, 1, 1, 1], padding='VALID')
            L2 = tf.nn.relu(L2)
            self.L2_flat = tf.reshape(L2, [-1, L2.shape[1] * L2.shape[2] * L2.shape[3]])

        W1 = tf.get_variable("W1", shape = [self.L2_flat.shape[1], self._hidden_size], initializer = tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable("b1", shape = [self._hidden_size])
        self._net = tf.nn.relu(tf.matmul(self.L2_flat, W1) + b1)

    # [Critic]
    def _BulidCriticNetwork(self):
        self._value = tf.contrib.layers.fully_connected(self._net, 1, activation_fn=None, scope="critic_output")        

        self._critic_loss = tf.reduce_mean(tf.squared_difference(self._value, self._reward))
        self._critic_train = tf.train.AdamOptimizer(self._critic_learning_rate).minimize(self._critic_loss)

    def CriticPredict(self, obs):
        p = self._sess.run([self._value], feed_dict={self._obs: obs})
        return p

    def CriticTrain(self, obs, rew):
        _, loss = self._sess.run([self._critic_train, self._critic_loss], feed_dict={self._obs : obs, self._reward : rew})
        print("Critic Loss :", loss);

    # [Actor]
    def _BulidActorNetwork(self):
        self._logit = tf.contrib.layers.fully_connected(self._net, self._output_size, activation_fn=tf.nn.softmax, scope="actor_output")
        log_p = -self._action * tf.log(tf.clip_by_value(self._logit, EPSILON, 1.))

        entropy = - self._logit * tf.log(tf.clip_by_value(self._logit, EPSILON, 1.))
        #entropy = tf.reduce_sum(entropy)
        
        log_lik = (log_p * self._advantage_value) + entropy * 0.001
        self._actor_loss = tf.reduce_mean(tf.reduce_sum(log_lik, axis=1))
        self._actor_train = tf.train.AdamOptimizer(self._actor_learning_rate).minimize(self._actor_loss)
    
    def GetAction(self, obs):
        obs = np.reshape(obs, [-1, self._height_size, self._width_size, self._depth_size])
        action = self._sess.run([self._logit], feed_dict={self._obs : obs})[0]
        action = np.random.choice(np.arange(self._output_size), p=action[0])
        return action

    def ActorTrain(self, obs, act, a_rew):
        _, loss = self._sess.run([self._actor_train, self._actor_loss], feed_dict={self._obs : obs, self._action : act, self._advantage_value : a_rew})
        print("Actor Loss :", loss);

    def _SetSaver(self):
        self._saver.CheckRestore()

    def Save(self, episode):
        if episode != 0 and episode % EPISODE_INTERVAL == 0:
            print("SAVE :", episode)
            self._saver.Save(episode)

def Server():
    global OBS_LIST
    global ACT_LIST
    global REW_LIST
    global ACTION
    global STATE

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:

        sock.bind((HOST, PORT))
        print("Server Waiting...")

        sock.listen(1)
        conn, addr = sock.accept()
        print("===[Client Accept]===")

        while True:
            if STATE == "state_wait":
                msg = conn.recv(BUFF_SIZE)
                msg = msg.decode('utf-8')
                try:
                    json_data = json.loads(msg)
                except:
                    print("Error!!!")
                    print(msg)

                type = json_data["type"]
                if type == "action":
                    data_list = json_data["datalist"]
                    for data in data_list:

                        stack = np.zeros(INPUT_SIZE, dtype = np.float)
                        for i in range(INPUT_SIZE):
                            key = "ob_" + str(i)
                            ob = data[key]
                            stack[i] = ob;

                        stack = np.reshape(stack, [1, HEIGHT_SIZE, WIDTH_SIZE, DEPTH_SIZE])
                        OBS_LIST.append(stack)

                    OBS_LIST = np.vstack(OBS_LIST)
                    STATE = "state_action"
                    
                elif type == "train":
                    data_list = json_data["datalist"]                
                    size = json_data["size"]
                    for data in data_list:

                        stack =  np.zeros(INPUT_SIZE, dtype = np.float)
                        for i in range(INPUT_SIZE):
                            key = "ob_" + str(i)
                            ob = data[key]
                            stack[i] = ob;

                        stack = np.reshape(stack, [1, HEIGHT_SIZE, WIDTH_SIZE, DEPTH_SIZE])
                        OBS_LIST.append(stack)

                        action = data["action"]
                        action = OneHot(int(action))
                        ACT_LIST.append(action)

                        reward = data["reward"]
                        REW_LIST.append(reward)

                    if len(OBS_LIST) == int(size):
                        OBS_LIST = np.vstack(OBS_LIST)
                        ACT_LIST = np.vstack(ACT_LIST)
                        REW_LIST = np.vstack(REW_LIST)
                        STATE = "state_train"

            elif STATE == "state_action_result":
                result = json.dumps({"result" : "action", "value" : str(ACTION)})
                msg = bytes(result, 'utf-8')
                conn.sendall(msg)

                STATE = "state_wait"

            elif STATE == "state_train_result":
                result = json.dumps({"result" : "train"})
                msg = bytes(result, 'utf-8')
                conn.sendall(msg)

                STATE = "state_wait"

        conn.close()

def Network():
    global EPISODE
    global OBS_LIST
    global ACT_LIST
    global REW_LIST
    global ACTION
    global STATE

    with tf.Session() as sess:
        network = ActorCriticNetwork(sess, HEIGHT_SIZE, WIDTH_SIZE, DEPTH_SIZE, OUTPUT_SIZE, PROJECT_NAME)
        while True:
            if STATE == "state_action":
                ACTION = network.GetAction(OBS_LIST)

                OBS_LIST = []
                STATE = "state_action_result"

            elif STATE == "state_train":
                REW_LIST = DiscountRewards(REW_LIST)
                predict = network.CriticPredict(OBS_LIST)
                new_reward = NormalizeRewards(REW_LIST, predict)

                network.CriticTrain(OBS_LIST, REW_LIST)
                network.ActorTrain(OBS_LIST, ACT_LIST, new_reward)

                EPISODE += 1
                network.Save(EPISODE)

                OBS_LIST = []
                ACT_LIST = []
                REW_LIST = []
                STATE = "state_train_result"


def OneHot(value):
    zero = np.zeros(OUTPUT_SIZE, dtype = np.int)
    zero[value] = 1
    return  zero

def DiscountRewards(reward_memory):
    v_memory = np.vstack(reward_memory)
    discounted = np.zeros_like(v_memory, dtype=np.float32)
    add_value = 0
    length = len(reward_memory)

    for i in reversed(range(length)):
        if v_memory[i] < 0:
            add_value = 0
        add_value = v_memory[i] + (DISCOUNT * add_value)
        discounted[i] = add_value

    return discounted

def NormalizeRewards(rewards, v_rewards):
    a_reward = np.vstack(rewards) - np.vstack(v_rewards)
    a_reward -= np.mean(a_reward)
    a_reward /= (np.std(a_reward) + EPSILON)
    return a_reward

def main():
    threading.Thread(target=Server).start()
    threading.Thread(target=Network).start()

if __name__ == "__main__":
    main()