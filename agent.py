import numpy as np
import utils

class Agent:    
    def __init__(self, actions, Ne=40, C=40, gamma=0.7):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path,self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None
    
    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        s_prime = self.generate_state(environment)

        # TODO: write your function here
        if (self._train != None):
            if (self.a != None) and (self.s != None):
                reward = -0.1
                if dead:
                    reward = -1
                elif points > self.points:
                    reward = 1

                self.N[self.s][self.a] += 1
                
                maxQ = self.Q[s_prime][0]

                for i in self.actions:
                    if maxQ < self.Q[s_prime][i]:
                        maxQ = self.Q[s_prime][i]

                learn_rate = self.C / (self.C + self.N[self.s][self.a])
                self.Q[self.s][self.a] += learn_rate * (reward + self.gamma * maxQ - self.Q[self.s][self.a])
            
            if dead:
                self.reset()
                return 0

            best = -10000
            curr = 0
            action = 0
            for i in reversed(self.actions):
                n_table = self.N[s_prime][i]
                if n_table < self.Ne:
                    curr = 1
                else:
                    q_table = self.Q[s_prime][i]
                    if q_table >= best:
                        curr = q_table
                if curr > best:
                    best = curr
                    action = i
        
            self.s = s_prime
            self.points = points
            self.a = action
            return self.a
        
        self.a = np.argmax(self.Q[s_prime])
        return self.a

    def generate_state(self, environment):
        # TODO: Implement this helper function that generates a state given an environment 
        snake_head_x, snake_head_y, snake_body, food_x, food_y = environment

        # food_dir_x and food_dir_y
        if snake_head_x == food_x:
            food_dir_x = 0
        elif snake_head_x > food_x:
            food_dir_x = 1
        else:
            food_dir_x = 2

        if snake_head_y == food_y:
            food_dir_y = 0
        elif snake_head_y > food_y:
            food_dir_y = 1
        else:
            food_dir_y = 2

        # adjoining_wall_x and adjoining_wall_y
        if snake_head_x == 1:
            adjoining_wall_x = 1
        elif snake_head_x == 16:
            adjoining_wall_x = 2
        else:
            adjoining_wall_x = 0 

        if snake_head_y == 1:
            adjoining_wall_y = 1
        elif snake_head_y == 8:
            adjoining_wall_y = 2
        else:
            adjoining_wall_y = 0

        # adjoining_body_top and adjoining_body_bottom and adjoining_body_left and adjoining_body_right
        adjoining_body_top = 0
        adjoining_body_bottom = 0
        adjoining_body_left = 0
        adjoining_body_right = 0

        for square in snake_body:
            if square[0] == snake_head_x and square[1] == snake_head_y - 1:
                adjoining_body_top = 1
            if square[0] == snake_head_x and square[1] == snake_head_y + 1:
                adjoining_body_bottom = 1
            if square[1] == snake_head_y and square[0] == snake_head_x - 1:
                adjoining_body_left = 1
            if square[1] == snake_head_y and square[0] == snake_head_x + 1:
                adjoining_body_right = 1

        return food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right