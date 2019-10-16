import gym
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image

resize = T.Compose([T.ToPILImage(),
                T.Resize(40, interpolation=Image.CUBIC),
                T.ToTensor()])

class GameInteraction(object):
    
    def __init__(self, env_id):
        # Create environnement
        self.env = gym.make(env_id).unwrapped
        self.env.reset()

    def get_screen(self):
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))
        screen = screen[:, :, :]
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        screen = resize(screen)
        return screen
    
    def act(self, action):
        _, reward, done, _ = self.env.step(action)
        return reward, done
    
    def reset(self):
        self.env.reset()
    

