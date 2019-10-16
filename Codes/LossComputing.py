import torch

class ComputeLoss(object):
    
    def __init__(self, hidden_lstm_size, gamma, screen_height, screen_width):
        self.hidden_lstm_size = hidden_lstm_size
        self.gamma = gamma
        self.screen_height = screen_height
        self.screen_width = screen_width
        
    def compute(self, experience, current_model, target_model):
        # import experiences to train on 
        state, action, next_state, reward, done = experience

        state_batch = torch.cat(state)
        state_batch = torch.FloatTensor(state_batch).unsqueeze(0)
        state_batch = state_batch.view([-1, 3, self.screen_width, self.screen_height])
        next_state = next_state[:len(next_state)-1] # we replace last one by 0 value
        next_state_batch = torch.cat(next_state)
        next_state_batch = torch.FloatTensor(next_state_batch).unsqueeze(0)
        next_state_batch = next_state_batch.view([-1, 3, self.screen_width, self.screen_height])
        reward_batch = torch.FloatTensor(reward)
        action_batch = torch.LongTensor(action)
        done = [1-d for d in done]
        done = torch.FloatTensor(done)
        
        # Initiate hidden layers for the current session (arbitrary choice)
        hidden_current = (torch.zeros(self.hidden_lstm_size).unsqueeze(0).unsqueeze(0), 
                 torch.zeros(self.hidden_lstm_size).unsqueeze(0).unsqueeze(0))
        hidden_target = (torch.zeros(self.hidden_lstm_size).unsqueeze(0).unsqueeze(0), 
                 torch.zeros(self.hidden_lstm_size).unsqueeze(0).unsqueeze(0))
        
        # Compute Q values 
        q_values, hidden_current = current_model(state_batch, hidden_current)
        q_value = q_values[0].gather(1, action_batch.reshape((-1, 1))).squeeze(1)
        next_q_values, hidden_target = target_model(next_state_batch, hidden_target) 
        next_q_value = next_q_values[0].max(1)[0]
        next_q_value = torch.cat([next_q_value, torch.Tensor([0])])
        
        expected_q_value = reward_batch + self.gamma * next_q_value * done
        
        # Compute loss
        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        return loss
    
