import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        # x: (batch, input_dim)
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)


def behavioral_clone(policy, X, y_target, epochs=5, lr=1e-3, device='cpu'):
    policy.to(device)
    opt = optim.Adam(policy.parameters(), lr=lr)
    X = _to_tensor(X, device)
    y = torch.tensor(y_target, dtype=torch.long, device=device)
    loss_fn = nn.CrossEntropyLoss()
    policy.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = policy.net(X)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()


def reinforce_train(policy, env, episodes=50, lr=1e-3, gamma=0.99, device='cpu', track_metrics=True):

    policy.to(device)
    opt = optim.Adam(policy.parameters(), lr=lr)
    policy.train()

    # Metrics containers
    episode_rewards = []
    episode_lengths = []
    episode_losses = []

    for ep in range(episodes):

        state = env.reset()

        log_probs = []
        rewards = []

        total_reward = 0
        steps = 0

        while True:

            st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            probs = policy(st)

            m = torch.distributions.Categorical(probs)

            action = m.sample()

            log_prob = m.log_prob(action)

            next_state, reward, done, _ = env.step(int(action.item()))

            log_probs.append(log_prob)
            rewards.append(reward)

            total_reward += reward
            steps += 1

            if done:
                break

            state = next_state

        # ----- Compute discounted returns -----

        returns = []
        R = 0.0

        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32, device=device)

        if returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # ----- Policy gradient loss -----

        loss = 0
        for lp, R in zip(log_probs, returns):
            loss = loss - lp * R

        opt.zero_grad()
        loss.backward()
        opt.step()

        # ----- Store metrics -----

        if track_metrics:
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            episode_losses.append(loss.item())

        print(f"Episode {ep+1}/{episodes} | Reward: {total_reward:.3f} | Steps: {steps} | Loss: {loss.item():.4f}")

    if track_metrics:
        return episode_rewards, episode_lengths, episode_losses


def policy_predict(policy, X, device='cpu'):
    policy.to(device)
    policy.eval()
    with torch.no_grad():
        Xt = _to_tensor(X, device)
        probs = policy(Xt).cpu().numpy()
        return np.argmax(probs, axis=1)


def _to_tensor(X, device='cpu'):
    try:
        arr = X.toarray()
    except Exception:
        arr = np.array(X)
    return torch.tensor(arr, dtype=torch.float32, device=device)
