#!/usr/bin/env python
#-*- coding:utf-8   -*-
#https://blog.csdn.net/wsc820508/article/details/82695870
#ROS开发笔记（10）——ROS 深度强化学习dqn应用之tensorflow版本(double dqn/dueling dqn/prioritized replay dqn)

import rospy
import random
import time
import tensorflow as tf
import numpy as np
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from std_msgs.msg import Float32MultiArray
 
# 导入 Env
from src.turtlebot3_dqn.environment_stage_1 import Env
 
# 采用莫烦强化学习教程中的实现
class SumTree(object):
    data_pointer = 0
    def __init__(self,capacity):
        # capacity 为回放经验的条数
        self.capacity=capacity
        # 初始化tree
        self.tree=np.zeros(2*capacity-1)
        # 初始化回放经验数据
        self.data=np.zeros(capacity,dtype=object)
    
    def add(self,p,data):
        # tree_idx 为所加data在树中的索引号
        self.tree_idx=self.data_pointer+self.capacity-1
        self.data[self.data_pointer]=data
        # 更新树
        self.update(self.tree_idx,p)
        self.data_pointer+=1
        if self.data_pointer>=self.capacity:
            self.data_pointer=0
 
    def update(self,tree_idx,p):
        change=p-self.tree[tree_idx]
        self.tree[tree_idx]=p
        # 更新tree的p
        while tree_idx!=0:
            # 更新父节点的p值
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
 
    def get_leaf(self,v):
        parent_idx = 0
        # 根据V寻找对应的叶子节点
        while True:    
            cl_idx = 2 * parent_idx + 1        
            cr_idx = cl_idx + 1
            # 判断是否达到树的底部
            if cl_idx >= len(self.tree):       
                leaf_idx = parent_idx
                break
            else:       
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx
 
        data_idx = leaf_idx - self.capacity + 1
        # 输出叶子节点序号，p值，以及对应的数据
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]
    
    @property
    def total_p(self):
        return self.tree[0] 
 
 
# 采用莫烦强化学习教程中的实现
class Memory(object):
    # 实际保存数据的条数
    saved_size=0
    epsilon = 0.01  # 避免 0 priority
    alpha = 0.6  # [0~1] 将 importance of TD error转化为 priority
    beta = 0.4  # importance-sampling 从这个初始值增加到1
    beta_increment_per_sampling = 0.001 
    abs_err_upper = 50.      
    def __init__(self,capacity):
        self.tree=SumTree(capacity)
 
    def store(self,transition):
        # 将新加入的transition 优先级p设为最高
        max_p=np.max(self.tree.tree[-self.tree.capacity:])
        if max_p==0:
            max_p=self.abs_err_upper
        self.tree.add(max_p,transition)
        if self.saved_size<self.tree.capacity:
            self.saved_size+=1
    
    def sample(self,n):
        # 初始化 b_idx, b_memory, ISWeights
        b_idx,  ISWeights = np.empty((n,), dtype=np.int32),  np.empty((n, 1))
        b_memory=[]
        self.beta=np.min([1.,self.beta+self.beta_increment_per_sampling])
        min_prob = np.min(self.tree.tree[self.tree.capacity-1:self.tree.capacity-1+self.saved_size]) / self.tree.total_p
        # print(self.tree.tree[self.tree.capacity-1:self.tree.capacity-1+self.saved_size]) 
 
        # 将total_p分为n份，每一份为pri_seg
        pri_seg = self.tree.total_p / n  
        for i in range(n):
            a,b= pri_seg*i, pri_seg*(i+1)
            v=random.uniform(a,b)
            idx, p, data=self.tree.get_leaf(v)
            prob=p/self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i]= idx
            b_memory.append(data)
        return b_idx, b_memory, ISWeights
 
    def batch_update(self,tree_idx,abs_errors):
        abs_errors += self.epsilon  # 避免0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
 
class DqnAgent():
    def __init__(self,
                 state_size,
                 action_size,
                 learning_rate = 0.00025,
                 drop_out=0.2,
                 discount_factor = 0.99,
                 epsilon = 1,
                 epsilon_decay=0.99,
                 epsilon_min=0.05,
                 episodes = 3000,
                 episode_step=6000,
                 target_update=2000,
                 memory_size = 1000000,
                 batch_size = 64,
                 output_graph = False,
                 summaries_dir="logs/",
                 sess=None,
                 double_dqn = True, 
                 prioritized=False,
                 dueling=True
                 ):
        self.dueling=dueling
        self.sess = sess
        self.prioritized=prioritized
        self.double_dqn = double_dqn
        # 创建 result 话题
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        # 初始化 result 话题
        self.result = Float32MultiArray()
 
        # 训练时的步数
        self.global_step = 0
 
        # 是否输出可视化图形,tensorboard相关
        self.output_graph=output_graph
 
 
        # 获取当前文件完整路径
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        # 基于当前路径生成模型保存路径前缀
        self.dirPath = self.dirPath+'/save_model/data_'
 
        # 导入前期训练的模型
        # self.load_model = True
        # self.load_episode = 150
        self.load_model = False
        self.load_episode = 0
 
        # 状态数
        self.state_size = state_size
        # 动作数
        self.action_size = action_size
 
        # 最大回合数
        self.episodes = episodes
 
        # 单个回合最大步数
        self.episode_step = episode_step
 
        # 每2000次更新一次target网络参数
        self.target_update=target_update
 
        # 折扣因子 计算reward时用 当下反馈最重要 时间越久的影响越小
        self.discount_factor=discount_factor
 
        # 学习率learning_rate  学习率决定了参数移动到最优值的速度快慢。
        # 如果学习率过大，很可能会越过最优值；反而如果学习率过小，优化的效率可能过低，长时间算法无法收敛。
        self.learning_rate=learning_rate
 
        # dropout层的rate
        self.drop_out=drop_out
 
        # 初始ϵ——epsilon
        # 探索与利用原则
        # 探索强调发掘环境中的更多信息，并不局限在已知的信息中；
        # 利用强调从已知的信息中最大化奖励；
        # greedy策略只注重了后者，没有涉及前者；
        # ϵ-greedy策略兼具了探索与利用，它以ϵ的概率从所有的action中随机抽取一个，以1−ϵ的概率抽取能获得最大化奖励的action。
        self.epsilon=epsilon
 
        # 随着模型的训练，已知的信息越来越可靠，epsilon应该逐步衰减
        self.epsilon_decay=epsilon_decay
 
        # 最小的epsilon_min，低于此值后不在利用epsilon_decay衰减
        self.epsilon_min=epsilon_min
 
        # batch_size 批处理大小
        # 合理范围内，增大 Batch_Size
        # 内存利用率提高了，大矩阵乘法的并行化效率提高
        # 跑完一次epoch（全数据集）所需要的迭代次数减小，对于相同数据量的处理速度进一步加快
        # 在一定范围内，一般来说batch size越大，其确定的下降方向越准，引起的训练震荡越小
 
        # 盲目增大batch size 有什么坏处
        # 内存利用率提高了，但是内存容量可能撑不住了
        # 跑完一次epoch（全数据集）所需要的迭代次数减少，但是想要达到相同的精度，其所花费的时间大大增加了，从而对参数的修正也就显得更加缓慢
        # batch size 大到一定的程度，其确定的下降方向已经基本不再变化
        self.batch_size = batch_size
 
        # 用于 experience replay 的 agent.memory
        # DQN的经验回放池(agent.memory)大于train_start才开始训练网络(agent.trainModel)
        self.train_start = self.batch_size
 
        # 用队列存储experience replay 数据，并设置队列最大长度
        if self.prioritized:
            self.memory=Memory(capacity=memory_size)
        else:
            self.memory=deque(maxlen=memory_size)
 
        # tensorboard保存路径
        self.summaries_dir = summaries_dir
 
        # 创建网络模型 [target_net, evaluate_net]
        self._buildModel()
 
        # 利用eval_net网络参数给target_net网络赋值
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
 
        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        
        if sess is None:
            self.sess = tf.Session()           
        else:
            self.sess = sess
 
        # tensorboard 可视化相关
        if self.output_graph:
            # 终端中执行 tensorboard --logdir=logs 命令，浏览器中输入 http://127.0.0.1:6006/ 可在tensorboard查看模型与数据
            # merged 也是一个操作，在训练时执行此操作
            self.merged = tf.summary.merge_all()
            # 创建 summary_writer
            self.summary_writer = tf.summary.FileWriter("logs/", self.sess.graph)
            print(self.summary_writer)
 
        # 2小时最多存5次
        self.saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)
 
        self.sess.run(tf.global_variables_initializer())
 
        # 训练可以加载之前保存的模型参数进行
        if self.load_model:
            self.saver.restore(self.sess, tf.train.latest_checkpoint('./checkpoint_dir/SaveModelDoubleDqnTf'))
 
            with open(self.dirPath + str(self.load_episode) + '.json') as outfile:
                param = json.load(outfile)
                self.epsilon = param.get('epsilon')
                # wsc self.epsilon = 0.5
                self.epsilon = 0.5
 
 
    def _buildModel(self):
 
        # ------------------ all inputs -----------------------------
        self.s = tf.placeholder(tf.float32, shape=[None, self.state_size], name='s')
        self.s_ = tf.placeholder(tf.float32, shape=[None, self.state_size], name='s_')
        self.q_target = tf.placeholder(tf.float32, [None, self.action_size], name='q_target')  
 
        if self.prioritized:
            self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
 
        # self.a = tf.placeholder(tf.int32, shape=[None, ], name='a')
 
        # net_config
        w_initializer = tf.random_normal_initializer(0., 0.3) 
        b_initializer = tf.constant_initializer(0.1)
 
        # ------------------ built evaluate_net ---------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 64, activation=tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            e2 = tf.layers.dense(e1, 64, activation=tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e2')
            e3 = tf.layers.dropout(e2, rate=self.drop_out, name='eval_drop_out')
            # self.q_eval = tf.layers.dense(e3, self.action_size,activation=tf.nn.softmax,kernel_initializer=w_initializer,
            #                               bias_initializer=b_initializer, name='q')
 
            if self.dueling:
                with tf.variable_scope('value'):
                    self.base_value = tf.layers.dense(e3, 1,kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='value')
                with tf.variable_scope('advantage'):
                    self.advantage_value = tf.layers.dense(e3, self.action_size,kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='advantage')
                with tf.variable_scope('Q'):
                    self.q_eval=self.base_value+(self.advantage_value-tf.reduce_mean(self.advantage_value, axis=1, keep_dims=True))
            else:
                self.q_eval = tf.layers.dense(e3, self.action_size,kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')        
        
        with tf.variable_scope('loss'):
            if self.prioritized:
                self.abs_errors = tf.reduce_sum(tf.abs(self.q_target - self.q_eval), axis=1)    # for updating Sumtree
                self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q_eval,name='TD_error'))
            else:
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval,name='TD_error'))
        
        # 添加scalar类型summary
        tf.summary.scalar('loss', self.loss)
 
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
 
 
        # ------------------ built target_net -----------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 64, activation=tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            t2 = tf.layers.dense(t1, 64, activation=tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t2')
            t3 = tf.layers.dropout(t2, rate=self.drop_out, name='target_drop_out')
 
            if self.dueling:
                with tf.variable_scope('t_value'):
                    self.t_base_value = tf.layers.dense(t3, 1,kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t_value')
                with tf.variable_scope('t_advantage'):
                    self.t_advantage_value = tf.layers.dense(t3, self.action_size,kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t_advantage')
                with tf.variable_scope('t_Q'):
                    self.q_next=self.t_base_value+(self.t_advantage_value-tf.reduce_mean(self.t_advantage_value, axis=1, keep_dims=True))
 
            else:
                # self.q_next = tf.layers.dense(t3, self.action_size, activation=tf.nn.softmax,
                #                               kernel_initializer=w_initializer,
                #                               bias_initializer=b_initializer, name='t3')
                self.q_next = tf.layers.dense(t3, self.action_size, 
                                            kernel_initializer=w_initializer,
                                            bias_initializer=b_initializer, name='t3')
    # 训练网络模型
    def trainModel(self, target=False):
        # 每经过target_update，更新target网络参数
        if self.global_step % self.target_update == 0:
            self.updateTargetModel()
            rospy.loginfo("UPDATE TARGET NETWORK")
            print("UPDATE TARGET NETWORK")           
 
        if self.prioritized:
            tree_idx, mini_batch, ISWeights = self.memory.sample(self.batch_size)
        else:
            mini_batch = random.sample(self.memory, self.batch_size)
 
 
        state_batch = np.empty((0, self.state_size), dtype=np.float64)
        action_batch = np.empty((0, ), dtype=np.float64)
        reward_batch = np.empty((0, ), dtype=np.float64)
        state_next_batch = np.empty((0, self.state_size), dtype=np.float64)
        q_target_batch=np.empty((0, self.action_size), dtype=np.float64)
 
        for i in range(self.batch_size):
            # states: [-0.25  0.  ]
            # states.reshape(1, len(states)):[[-0.5  0. ]] ,一行数据
            # print(mini_batch[i]) (array([-0.5,  0. ]), 2, 0, array([-0.25,  0.  ]), False)
            state = mini_batch[i][0]
            action = mini_batch[i][1]
            reward = mini_batch[i][2]
            next_state = mini_batch[i][3]
            # done = mini_batch[i][4]
 
            # state=state.reshape(1,len(state))
            # next_state=next_state.reshape(1,len(next_state))
 
            # 将next_state 分别送入 target_net 与 eval_net 分别获得 q_next, q_eval_next
            q_next, q_eval_next = self.sess.run(
                [self.q_next, self.q_eval],
                feed_dict={self.s_: [next_state], self.s: [next_state]})  
 
            # 将 state 送入  eval_net  获得  q_eval
            q_eval = self.sess.run(self.q_eval, {self.s: [state]})
 
            q_target = q_eval.copy()
 
            if self.double_dqn:
                # double DQN
                # 选择 q_eval_next 向量中最大值对应的动作序号
                max_act_next_state = np.argmax(q_eval_next, axis=1) 
                # Double DQN, 根据动作序号选择 q_next 值
 
                selected_q_next = q_next[0,max_act_next_state]  
            else:
                 # DQN
                selected_q_next = np.max(q_next, axis=1)   
            
            q_target[0,action] = reward + self.discount_factor * selected_q_next
 
            state_batch = np.append(state_batch, np.array([state.copy()]), axis=0)
 
            q_target_batch=np.append(q_target_batch, np.array(q_target.copy()), axis=0)
                       
            action_batch = np.append(action_batch, np.array([action]), axis=0)
            reward_batch = np.append(reward_batch, np.array([reward]), axis=0)
            state_next_batch = np.append(state_next_batch, np.array([next_state.copy()]), axis=0)       
        
        # tensorboard 可视化相关
        if self.output_graph:
            if self.prioritized:
                summary,_, abs_errors, self.cost ,self.q_value= self.sess.run([self.merged, self._train_op, self.abs_errors, self.loss,self.q_eval],
                                            feed_dict={self.s: state_batch,
                                                        self.q_target: q_target_batch,
                                                        self.ISWeights: ISWeights})
                self.memory.batch_update(tree_idx, abs_errors)    
            else:
                # 这里运行了self.merged操作
                summary, _,self.q_value = self.sess.run([self.merged, self._train_op,self.q_eval],
                                        feed_dict={
                                            self.s: state_batch,
                                            self.q_target: q_target_batch
                                        })
                # 保存 summary
                self.summary_writer.add_summary(summary, self.global_step)
        else:
             if self.prioritized:
                _, abs_errors, self.cost,self.q_value = self.sess.run([self._train_op, self.abs_errors, self.loss,self.q_eval],
                                            feed_dict={self.s: state_batch,
                                                        self.q_target: q_target_batch,
                                                        self.ISWeights: ISWeights})
                self.memory.batch_update(tree_idx, abs_errors)  
             else:
                 _ ,self.q_value= self.sess.run([self._train_op,self.q_eval],
                                        feed_dict={
                                            self.s: state_batch,
                                            self.q_target: q_target_batch
                                        })
 
    def updateTargetModel(self):
        self.sess.run(self.target_replace_op)
        print('\ntarget_params_replaced\n')
 
    def appendMemory(self, state, action, reward, next_state, done):
        if self.prioritized:    
            transition = (state, action, reward, next_state)
            self.memory.store(transition)    
        else:       
            self.memory.append((state, action, reward, next_state, done))
 
 
    # 基于ϵ——epsilon策略选择动作
    def chooseAction(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
            return action
 
        else:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: [state]})
            self.q_value = actions_value
            action = np.argmax(actions_value)
            return action
 
 
if __name__ == '__main__':
 
    rospy.init_node('turtlebot3_dqn_stage_1_tensorflow')
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    result = Float32MultiArray()
    get_action = Float32MultiArray()
 
    state_size = 26
    action_size = 5
 
    env = Env(action_size)
 
    agent = DqnAgent(state_size, action_size,output_graph=True)
    scores, episodes = [], []
    agent.global_step = 0
    start_time = time.time()
 
    # 循环EPISODES个回合
    for e in range(agent.load_episode + 1, agent.episodes):
        done = False
        state = env.reset()
        score = 0
 
        # 每10个回合保存一次网络模型参数
        if e % 10 == 0:
            # 保存参数,不保存graph结构
            agent.saver.save(agent.sess, "./checkpoint_dir/SaveModelDoubleDqnTf", global_step=e, write_meta_graph=True)
            with open(agent.dirPath + str(e) + '.json', 'w') as outfile:
                param_keys = ['epsilon']
                param_values = [agent.epsilon]
                param_dictionary = dict(zip(param_keys, param_values))
                json.dump(param_dictionary, outfile)
                print('epsilon saver')
 
        # 每个回合循环episode_step步
        for t in range(agent.episode_step):
 
            # 选择动作
            action = agent.chooseAction(state)
            # Env动作一步，返回next_state, reward, done
            next_state, reward, done = env.step(action)
            # print(reward)
            # 存经验值
            agent.appendMemory(state, action, reward, next_state, done)
 
            # agent.memory至少要收集agent.train_start（64）个才能开始训练
            # agent.global_step 没有达到 agent.target_update之前要用到target网络的地方由eval代替
            if agent.global_step >= agent.train_start:
                if agent.global_step <= agent.target_update:
                    agent.trainModel()
                else:
                    agent.trainModel(True)
            # 将回报值累加成score
            score += reward
 
            state = next_state
            # 发布 get_action 话题
            get_action.data = [action, score, reward]
            pub_get_action.publish(get_action)            
 
            # 超过500步时设定为超时，回合结束
            if t >= 500:
                print("Time out!!")
                done = True
 
            if done:
                # 发布result话题
                result.data = [score, np.max(agent.q_value)]
                pub_result.publish(result)
 
                scores.append(score)
                episodes.append(e)
                # 计算运行时间
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)
 
                rospy.loginfo('Ep: %d score: %.2f global_step: %d epsilon: %.2f time: %d:%02d:%02d',e, score, agent.global_step, agent.epsilon, h, m, s)
 
                break
 
            agent.global_step += 1
 
        # 更新衰减epsilon值，直到低于等于agent.epsilon_min
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

