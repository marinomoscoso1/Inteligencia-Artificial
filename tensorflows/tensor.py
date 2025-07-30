import tensorflow as tf
import numpy as np
import copy
import sys
from tensorflows.structure_creator import create_structure

class Envirement:
    def __init__(self,file):
        self.file=file

        self.original_structure,self.agent_position,self.goal_position=create_structure(self.file)

        self.structure=copy.deepcopy(self.original_structure)

        self.actual_agent_position=self.agent_position[:]

    def reset(self):
        self.structure=copy.deepcopy(self.original_structure)

        self.actual_agent_position=self.agent_position[:]

        return self.actual_agent_position
    
    def step(self,action):
        reward_goal=100
        reward_obsta=-10
        reward_pass=-1
        invalid_reward=-8

        match action:
            case 0:
                new_state=[self.actual_agent_position[0],self.actual_agent_position[1]-1]
            case 1:
                new_state=[self.actual_agent_position[0],self.actual_agent_position[1]+1]
            case 2:
                new_state=[self.actual_agent_position[0]+1,self.actual_agent_position[1]]
            case 3:
                new_state=[self.actual_agent_position[0]-1,self.actual_agent_position[1]]
    
        if new_state[0]<0 or new_state[1]<0 or new_state[0]>=len(self.structure) or new_state[1]>=len(self.structure[0]):
            return invalid_reward,new_state,False
    
        cell= self.structure[new_state[0]][new_state[1]]

        if cell=="G":
            self.structure[new_state[0]][new_state[1]]="A"
            self.actual_agent_position=new_state
            return reward_goal,new_state,True
        elif cell=="#":
            return reward_obsta,new_state,False
        elif cell==".":
            self.structure[new_state[0]][new_state[1]]="A"
            self.structure[self.actual_agent_position[0]][self.actual_agent_position[1]]="."
            self.actual_agent_position=new_state
            for i in self.structure:
                print(i)

        return reward_pass,self.actual_agent_position,False
    
    def get_state(self):

        filas,columnas=len(self.structure),len(self.structure[0])

        encoding_structure=np.zeros((filas,columnas,4),dtype=np.float32)

        for i in range(filas):
            for j in range(columnas):
                c=self.structure[i][j]

                match c:
                    case "#":
                        encoding_structure[i][j]=[1,0,0,0]
                    case ".":
                        encoding_structure[i][j]=[0,1,0,0]
                    case "A":
                        encoding_structure[i][j]=[0,0,1,0]
                    case "G":
                        encoding_structure[i][j]=[0,0,0,1]

        return encoding_structure

    def render(self,structure):
        for i in self.structure:
            print(i)

class Agent:

    def __init__(self):
        self.model=tf.keras.Sequential()
        self.model.add(tf.keras.layers.Conv2D)

        




