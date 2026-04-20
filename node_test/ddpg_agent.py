import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import copy

class OUNoise:
    """Ornstein-Uhlenbeck 过程，用于生成时间相关的平滑噪声，增强连续动作空间的探索能力"""
    def __init__(self, action_dimension, scale=0.1, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1) # 保证输出总和为1
        )

    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


class DDPGAgent:
    def __init__(self, state_dim, action_dim, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.005):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)

        self.actor_target = Actor(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.gamma = gamma
        self.tau = tau
        
        # 基础 DDPG 的 OU 噪声
        self.ou_noise = OUNoise(action_dimension=action_dim)

    def select_action(self, state, epsilon=0.1):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.actor(state_tensor).detach().numpy()[0]
        
        # 加入 OU 噪声并重新归一化 (epsilon 作为噪声开关或缩放)
        if epsilon > 0:
            noise = self.ou_noise.noise() * epsilon
            action = probs + noise
            action = np.clip(action, 1e-5, 1.0)
            action = action / action.sum()
        else:
            action = probs
            
        return action

    def select_deterministic_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.actor(state_tensor).detach().numpy()[0]
        return probs / probs.sum()

    def add_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_value = rewards + (1 - dones) * self.gamma * target_q

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_value)

        # Actor 试图最大化 Critic 评价的 Q 值，因此加负号作为 Loss
        policy_loss = -self.critic(states, self.actor(states)).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.actor_target, self.actor, self.tau)
        self._soft_update(self.critic_target, self.critic, self.tau)

    def _soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filepath):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])


class DirichletActor(nn.Module):
    def __init__(self, state_dim, n_nodes, hidden_dim=256):
        super().__init__()
        self.n_nodes = n_nodes
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_nodes),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.net(state)


class DDPGAgentDirichlet(DDPGAgent):
    def __init__(self, state_dim, n_nodes, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.005):
        super().__init__(state_dim, n_nodes, lr_actor, lr_critic, gamma, tau)

        self.actor = DirichletActor(state_dim, n_nodes)
        self.actor_target = DirichletActor(state_dim, n_nodes)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.ou_noise = OUNoise(action_dimension=n_nodes)

    def select_action(self, state, epsilon=0.1):
        """复现论文公式 (16): mu(s_t) = Dir(theta_t) + [mu(s_t|theta) + N]"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        # 获取 Actor 网络的输出基准概率
        probs = self.actor(state_tensor).detach().numpy()[0]
        
        if epsilon > 0:
            # 1. 结合 Dirichlet 分布特性进行采样
            # 放大 probs 作为 Dirichlet 的 alpha 参数，以 Actor 输出为分布中心
            alpha = probs * 10.0 + 1.0 
            dirichlet_action = np.random.dirichlet(alpha)
            
            # 2. 引入 OU 噪声
            noise = self.ou_noise.noise() * epsilon
            
            # 3. 融合并确保动作合法 (非负且和为1)
            action = dirichlet_action + noise
            action = np.clip(action, 1e-5, 1.0)
            action = action / action.sum()
        else:
            action = probs / probs.sum()
            
        return action