# Observe your machine learning to play video game

## Installation

You can directly _clone_ the repository and install the packages listed in the `requirements.txt` file. 
Here are the commands with python in 3.6.5 :
```
git clone https://github.com/VBrabant/Observe_ReinforcementLearning
pip install -r requirements.txt
```

##  Description

**This application of Reinforcement Learning (Markov Decision Process) is using features like Dueling-DQN, POMDP, and experience replay to allow you to watch your computer learning to play the _CartPole-v0_ game in the Gym library.**

You can choose how many games you want your computer to play with the [-n] option. By default it will play 1500 games.
example :
```
python Codes/Training.py -n 2000
```