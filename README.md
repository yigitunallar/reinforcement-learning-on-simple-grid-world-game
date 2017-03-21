# Reinforcement Learning on Simple Games
## Introduction
Learning from interaction is a foundational idea underlying nearly all theories of learning and intelligence. Driving a car, holding a conversation might be the most essentials of this domain [1]. Reinforcement learning is much more focused on goal-directed learning from interaction than are other approaches to machine learning. One might be tempted to think of reinforcement learning as a kind of unsupervised learning because it does not rely on examples of correct behavior, but reinforcement learning is trying to maximize a reward signal instead of trying to find hidden structure [2]. Recent advances in deep learning have made it possible to extract high-level features from raw sensory data, leading to breakthroughs in computer vision and speech recognition. These methods utilize neural network architectures, including convolutional networks, recurrent neural networks etc. [3] It seems natural to ask whether the similar techniques could also be beneficial for RL. This paper demonstrates that a neural network can be utilized to learn successful control policies in RL environments. The network is trained with a variant of the Q-learning algorithm. To alleviate the problems of correlated data and non-stationary distributions, we use an experience replay mechanism [4] which randomly samples previous transitions, and thereby smooths the training distribution over many past behaviors. We apply our approach to a simple 4x4 grid game, where we will create a single neural network agent that is able to successfully learn to play this game. The game environment and the network structure will be detailed in the following sections.
## Grid Game
 It is a simple text based game in which there is a 4X4 grid of tiles and 4 objects placed. The objects and the agent’s movement domains are as follows:
* Objects,
  * Agent,
  * Pit,
  * Goal,
  * Wall,
* Movements,
  * Up,
  * Down,
  * Left,
  * Right,
  
The ultimate point of the game is to get to the goal where the player will receive a numerical reward. We also have to avoid a pit, in order not to get negative reward. Additionally, although it offers no reward or penalty, we have to avoid the wall that blocks the agent’s path. The grid environment is shown below: 

![picture1](https://cloud.githubusercontent.com/assets/18366839/24140344/5590a56e-0e28-11e7-837f-2209c3386ce1.png)

This implementation does not sound very exciting, as the game itself does not teach the agent how to play a hard game at a super human level. However, it actually uses many of the same principles used by DeepMind’s Atari playing algorithm [3]. In the next parts, we will often be using the below terms and concepts, thus reviewing them once again will help making this study more self-contained. 
Policy, denoted Π, is the specific strategy we take in order to get into high value states or take high value actions to maximize our rewards over time. Generally, Π(s) as a function just evaluates the value of all possible actions given the state s and returns the highest value action. 
Value function, that accepts a state s, and returns the value of the state vπ(s).
Action-Value function, Q (s, a ), that accepts a state s, and an action a and returns the value of taking that action given that state.

## Q-Learning
It is a type of algorithm used to calculate state-action values. It is a member of TD algorithms, which indicates that, time differences between actions taken and rewards received are involved. With TD algorithms, we make updates after every action taken, we basically make a prediction, take an action based on that prediction, receive a reward and then update our prediction. The tabular Q learning update rule is as follows [5]:
> Q (St, At)← Q(St, At)+α[Rt+1+γmaxQ(St+1, at+1)−Q(St, At)]

Our Q (s, a) function does not have to be a lookup table. In most problems, state-action space is much too large to store in a table. What we need is some way to generalize and pattern match between states. This is exactly where neural networks come in. We can use a neural network, instead of a lookup table, as our Q(s ,a) function. It will simply accept a state and an action and spit out the value of that state-action. As for the neural network weights, our Q function will actually look like this: Q (s, a, theta). Instead of iteratively updating values in a table, we will iteratively update the theta parameters of our neural network so that it learns to provide us with better estimates of state-action values. Since the neural network is not a table, we do not use the formula shown above. For any state-action that just happened, our target would be,
> rt+1+γ∗ maxQ(s′,a′)

γ is a parameter 0 -> 1, that is called discount factor. It determines how much each future reward is taken into consideration for updating our Q-values. s′ and a′ are used interchangeably with s t+1 and a t+1. Except for the terminal state, our reward update is carried out using the formula above. As for the terminal state, the reward update is simply r t+1. There are 2 terminal states; the state where the agent fell into the pit and receives -10, and the state where the player has reached the goal and receives +10.

### On-policy vs. Off-Policy
On-policy methods iteratively learn about state values and improve its policy simultaneously, the updates to state values depending on our policy. On the contrary, off-policy methods (e.g. Q-learning) allow the agent to follow one policy, while learning about another. For instance, the agent can take completely random actions and he could still learn about another policy function of taking the best actions in every state. 

## Neural Network as Q Function
We are here using the fairly popular Theano-based library Keras [6]. As Google’s DeepMind did for its Atari playing algorithm, we will build a network that just accepts a state and outputs separate Q-values for each possible action in its output layer. This is clever, because rather than running a NN for every action, we just need to run it forward once. 

![picture2](https://cloud.githubusercontent.com/assets/18366839/24140343/553e908a-0e28-11e7-84f4-e34eddd4e7a8.png)

The NN is structured as shown in Figure 2 in 4 layers; input layer of 64 units (4x4x4 array), 2 hidden layers of 164 and 150 units, and an output layer of 4, one for each of our possible actions (up, down, left, right). After having done many trial and errors, we came up with this layer architectures that seemed to work fairly well. Another researcher may change it up and get better results.

## Grid World Details
In this section, we will discuss the game implementation that we are using as a learning problem. Basically, the goal, pit and the wall will always be initialized in the same positions, but the player will be placed randomly on the grid on each new game. The state is a 3-dimensional numpy array(4x4x4). First 2 dimensions are the positions on the board. The 3rd dimension encodes the object/element at that position. Since there are 4 different possible objects, the 3rd dimension of the state contains vectors of length 4. With a 4 length vector we are encoding 5 possible options at each grid position: empty, player, goal, pit, or wall. We check if we can find “clean” arrays-only one “1” in the “Z” dimension of a particular grid position of the various element types on the grid and if not, we just recursively call the initialize grid function until we get a state where elements are not superimposed. When the player successfully plays the game and lands on the goal, the player and goal positions will be superimposed and that is how we know the player has won. The wall is supposed to block the movement of the player so we prevent the player form taking an action that would place them at the same position as the wall. Additionally, the grid is “enclosed” so that player cannot walk through the edges of the grid. The first thing we do is try to find the positions of each element on the grid. Then it is just a few simple if-conditions. We need to make sure the player isn’t trying to step on the wall and make sure that the player isn’t stepping outside the bounds of the grid. Additionally, we will implement our reward function, which will award +10 if the player steps onto the goal, -10 if the player steps into the pit, and -1 for any other move. These rewards are pretty arbitrary, as long as the goal has a significantly higher reward than the pit, the algorithm should do fine. Implemented function in the application code will display our grid as a text array, so we can observe each movement on the game space individually. The player and pit or the goal disappear when they are superimposed. 

## Training the Network
As discussed before, we set up our network using keras library as per the devised layer structure, and now it is required to train it to yield learned results. In case the agent is randomly placed on the grid, we have to implement experience replay, which gives us minibatch updating in an online learning scheme. To implement and train our network, we have to follow the steps below: 
*	In state s, take action a, observe new state st+1, and reward rt+1,
*	Store this as a tuple (s,a,st+1, rt+1) in a list,
*	Continue to store each experience in this list until we have filled the list to a specific length,
*	Once the experience replay memory is filled, randomly select a subset,
*	Iterate through this subset and calculate value updates for each; store these in a target array y_train and store the state s each memory in x_train,
*	Use x_train and y_train as a minibatch for batch training. For subsequent epochs (a full game played to completion) where the array is full, just overwrite old values in our experience replay memory array.

In addition to learning the action-value for the action we just took, we are also going to use a random sample of our past experiences to train on to prevent catastrophic forgetting. Having resolved the catastrophic forgetting, we simply implement our algorithm as follows: 
*	Setup a for-loop to number of epochs,
*	Setup a while loop,
*	Run Q network forward,
*	Implement epsilon greedy implementation, at time t with probability epsilon we will choose a random action. With probability 1-epsilon  we will choose the action associated with the highest Q value from our neural network,
*	Implement experience replay method described above, 
*	Train the model on 1 sample. Repeat the processes above.

## Running the Code and Test Results
The easiest way to run the application is using Jupyter [7], which is a web application that allows the user to create and share documents that contain live code, equations etc. To test the code, the user needs to upload the “.ipynb” file to the Jupyter website, then each section of the code will be able to run easily. Firstly, we need to setup the grid world environment as shown below: 

```
import numpy as np

def randPair(s,e):
    return np.random.randint(s,e), np.random.randint(s,e)

#finds an array in the "depth" dimension of the grid
def findLoc(state, obj):
    for i in range(0,4):
        for j in range(0,4):
            if (state[i,j] == obj).all():
                return i,j

#Initialize player in random location, but keep wall, goal and pit stationary
def initGridPlayer():
    state = np.zeros((4,4,4))
    #place player
    state[randPair(0,4)] = np.array([0,0,0,1])
    #place wall
    state[2,2] = np.array([0,0,1,0])
    #place pit
    state[1,1] = np.array([0,1,0,0])
    #place goal
    state[1,2] = np.array([1,0,0,0])

    a = findLoc(state, np.array([0,0,0,1])) #find grid position of player (agent)
    w = findLoc(state, np.array([0,0,1,0])) #find wall
    g = findLoc(state, np.array([1,0,0,0])) #find goal
    p = findLoc(state, np.array([0,1,0,0])) #find pit
    if (not a or not w or not g or not p):
        #print('Invalid grid. Rebuilding..')
        return initGridPlayer()

    return state
```
As indicated above, we will randomly locate the player at each run of this code. Having implemented the environment successfully, we need to define movement functions.

```
def makeMove(state, action):
    #need to locate player in grid
    #need to determine what object (if any) is in the new grid spot the player is moving to
    player_loc = findLoc(state, np.array([0,0,0,1]))
    wall = findLoc(state, np.array([0,0,1,0]))
    goal = findLoc(state, np.array([1,0,0,0]))
    pit = findLoc(state, np.array([0,1,0,0]))
    state = np.zeros((4,4,4))

    #up (row - 1)
    if action==0:
        new_loc = (player_loc[0] - 1, player_loc[1])
        if (new_loc != wall):
            if ((np.array(new_loc) <= (3,3)).all() and (np.array(new_loc) >= (0,0)).all()):
                state[new_loc][3] = 1
    #down (row + 1)
    elif action==1:
        new_loc = (player_loc[0] + 1, player_loc[1])
        if (new_loc != wall):
            if ((np.array(new_loc) <= (3,3)).all() and (np.array(new_loc) >= (0,0)).all()):
                state[new_loc][3] = 1
    #left (column - 1)
    elif action==2:
        new_loc = (player_loc[0], player_loc[1] - 1)
        if (new_loc != wall):
            if ((np.array(new_loc) <= (3,3)).all() and (np.array(new_loc) >= (0,0)).all()):
                state[new_loc][3] = 1
    #right (column + 1)
    elif action==3:
        new_loc = (player_loc[0], player_loc[1] + 1)
        if (new_loc != wall):
            if ((np.array(new_loc) <= (3,3)).all() and (np.array(new_loc) >= (0,0)).all()):
                state[new_loc][3] = 1

    new_player_loc = findLoc(state, np.array([0,0,0,1]))
    if (not new_player_loc):
        state[player_loc] = np.array([0,0,0,1])
    #re-place pit
    state[pit][1] = 1
    #re-place wall
    state[wall][2] = 1
    #re-place goal
    state[goal][0] = 1

    return state
```
As shown above, movement actions are encoded as follows: 
*	0=up,
*	1=down,
*	2=left,
*	3=right.

We also made sure that the player is not trying to step on the wall and the player is not stepping outside the bounds of the grid. 
Next, we implement the location, and reward function which will award +10 if the player steps onto the goal, -10 if the player steps into the pit, and -1 for any other move. To display the grid, a function designed to print the grid in a text array. The implemented code is shown below.

```
def getLoc(state, level):
    for i in range(0,4):
        for j in range(0,4):
            if (state[i,j][level] == 1):
                return i,j

def getReward(state):
    player_loc = getLoc(state, 3)
    pit = getLoc(state, 1)
    goal = getLoc(state, 0)
    if (player_loc == pit):
        return -10
    elif (player_loc == goal):
        return 10
    else:
        return -1

def dispGrid(state):
    grid = np.zeros((4,4), dtype='<U2')
    player_loc = findLoc(state, np.array([0,0,0,1]))
    wall = findLoc(state, np.array([0,0,1,0]))
    goal = findLoc(state, np.array([1,0,0,0]))
    pit = findLoc(state, np.array([0,1,0,0]))
    for i in range(0,4):
        for j in range(0,4):
            grid[i,j] = ' '

    if player_loc:
        grid[player_loc] = 'P' #player
    if wall:
        grid[wall] = 'W' #wall
    if goal:
        grid[goal] = '+' #goal
    if pit:
        grid[pit] = '-' #pit

    return grid
```
So far, we have completed to implement our game world. Next, we need to setup our NN and RL algorithms, and test it against our game environment.

```
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

model = Sequential()
model.add(Dense(164, init='lecun_uniform', input_shape=(64,)))
model.add(Activation('relu'))

model.add(Dense(150, init='lecun_uniform'))
model.add(Activation('relu'))


model.add(Dense(4, init='lecun_uniform'))
model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)

model.predict(state.reshape(1,64), batch_size=1)
#just to show an example output; read outputs left to right: up/down/left/right
```

As we described in previous sections, our NN comprise of 4 layers, each one defined as separate lines in Keras library. To demonstrate, the last lines of code above calculates the example output of the NN using the state information. 
To train the network, we implemented the code below. We defined some parameters like epochs, gamma, batchSize and buffer empiracally. Another values might change the outcome greatly. 

```
model.compile(loss='mse', optimizer=rms)#reset weights of neural network
epochs = 3000
gamma = 0.975
epsilon = 1
batchSize = 40
buffer = 80
replay = []
#stores tuples of (S, A, R, S')
h = 0
for i in range(epochs):

    state = initGridPlayer() #using the harder state initialization function
    status = 1
    #while game still in progress
    while(status == 1):
        #We are in state S
        #Let's run our Q function on S to get Q values for all possible actions
        qval = model.predict(state.reshape(1,64), batch_size=1)
        if (random.random() < epsilon): #choose random action
            action = np.random.randint(0,4)
        else: #choose best action from Q(s,a) values
            action = (np.argmax(qval))
        #Take action, observe new state S'
        new_state = makeMove(state, action)
        #Observe reward
        reward = getReward(new_state)

        #Experience replay storage
        if (len(replay) < buffer): #if buffer not filled, add to it
            replay.append((state, action, reward, new_state))
        else: #if buffer full, overwrite old values
            if (h < (buffer-1)):
                h += 1
            else:
                h = 0
            replay[h] = (state, action, reward, new_state)
            #randomly sample our experience replay memory
            minibatch = random.sample(replay, batchSize)
            X_train = []
            y_train = []
            for memory in minibatch:
                #Get max_Q(S',a)
                old_state, action, reward, new_state = memory
                old_qval = model.predict(old_state.reshape(1,64), batch_size=1)
                newQ = model.predict(new_state.reshape(1,64), batch_size=1)
                maxQ = np.max(newQ)
                y = np.zeros((1,4))
                y[:] = old_qval[:]
                if reward == -1: #non-terminal state
                    update = (reward + (gamma * maxQ))
                else: #terminal state
                    update = reward
                y[0][action] = update
                X_train.append(old_state.reshape(64,))
                y_train.append(y.reshape(4,))

            X_train = np.array(X_train)
            y_train = np.array(y_train)
            print("Game #: %s" % (i,))
            model.fit(X_train, y_train, batch_size=batchSize, nb_epoch=1, verbose=1)
            state = new_state
        if reward != -1: #if reached terminal state, update game status
            status = 0
        clear_output(wait=True)
    if epsilon > 0.1: #decrement epsilon over time
        epsilon -= (1/epochs)
```

Lastly, we need to implement a test algorithm to initiate the game and run our learning algorithm against it. Below set-up displays the simple code snippet that sets up the environment and run NN.

```
def testAlgo(init):
    i = 0
    
    if init==1:
        state = initGridPlayer()
    
    print("Initial State:")
    print(dispGrid(state))
    status = 1
    #while game still in progress
    while(status == 1):
        qval = model.predict(state.reshape(1,64), batch_size=1)
        action = (np.argmax(qval)) #take action with highest Q-value
        print('Move #: %s; Taking action: %s' % (i, action))
        state = makeMove(state, action)
        print(dispGrid(state))
        reward = getReward(state)
        if reward != -1:
            status = 0
            print("Reward: %s" % (reward,))
        i += 1 #If we're taking more than 10 actions, just stop, we probably can't win this game
        if (i > 10):
            print("Game lost; too many moves.")
            break
```

Having completed the entire programming work, we can run the “testAlgo(1)” to see the performance of our learning algorithm. After many runs, a user might come across the typical result depicted below:

![picture3](https://cloud.githubusercontent.com/assets/18366839/24140346/564488ae-0e28-11e7-9b99-876369a2ab4a.png)

## Conclusion 
As depicted in the previous section, our algorithm successfully learned the way leading to reward (+) state in exactly 6 movements. Each run we may end up with different results as we let the randomness come in while initializing our player’s position. 
In this paper, we successfully proved that, Q-learning with NN learns pretty neatly, and using experience replay, we are able to combat against the random initialization of the player.
As for future work, we aim at implementing the algorithm in a more realistic scenario, initiating every single object of the game world randomly.

## References
* [1]	Sutton, R. S. and Barto, A. G. Introduction to reinforcement learning. MIT Press, 1998.
* [2]	Alpaydin, E. Introduction to Machine Learning. MIT Press, 2010.
* [3]	Volodymyr Mnih, Koray Kovukcuoglu, David Silver. Playing Atari with Deep Reinforcement Learning, 2013.
* [4]	Long-Ji Lin. Reinforcement learning for robots using neural networks. Technical report, DTICDocument, 1993.
* [5]	Marta Garnelo, Kai Arulkumaran, Murray Shanahan. Towards Deep Symbolic Reinforcement Learning, 2016.
* [6]	https://keras.io/
* [7]	https://jupyter.org/
