import tensorflow as tf
import numpy as np
import copy
import random
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from structure_creator import create_structure

class Enviroment:
    def __init__(self,file):
        self.file=file

        self.original_structure,self.agent_position,self.goal_position=create_structure(self.file)

        self.structure=copy.deepcopy(self.original_structure)

        self.actual_agent_position=self.agent_position[:]

        self.visited_positions=set()

        self.repetitive_penalty=-5
        self.goal_reward=100
        self.best_avg_reward=float("-inf")
        self.historical_rewards=[]

    def reset(self):
        self.structure=copy.deepcopy(self.original_structure)

        self.actual_agent_position=self.agent_position[:]

        self.visited_positions=set()

        return self.actual_agent_position

    def update_penalty_needed(self,episode,reward):
        self.historical_rewards.append(reward)

        if episode%50==0 and len(self.historical_rewards)>=50:
            last_data=self.historical_rewards[-50:]
            average=sum(last_data)/len(last_data)

            if average>self.best_avg_reward:
                self.best_avg_reward=average
            else:
                if self.repetitive_penalty>-12:
                    self.repetitive_penalty-=1
                if self.goal_reward<210:
                    self.goal_reward+=5
    
    def step(self,action):
        reward_obsta=-10
        reward_pass=1
        invalid_reward=-8

        print(action,"accion,modelo")


        match action:
            case 0:
                new_state=[self.actual_agent_position[0],self.actual_agent_position[1]-1] #izquierda
            case 1:
                new_state=[self.actual_agent_position[0],self.actual_agent_position[1]+1] #Derecha
            case 2:
                new_state=[self.actual_agent_position[0]+1,self.actual_agent_position[1]] #Abajo
            case 3:
                new_state=[self.actual_agent_position[0]-1,self.actual_agent_position[1]] #Arriba
    
        if new_state[0]<0 or new_state[1]<0 or new_state[0]>=len(self.structure) or new_state[1]>=len(self.structure[0]):
            print("camino invalido")
            return invalid_reward,new_state,False
        
        if tuple(new_state) in self.visited_positions:
            print("ojo que ya lo visito")
            self.structure[new_state[0]][new_state[1]]="A"
            self.structure[self.actual_agent_position[0]][self.actual_agent_position[1]]="."
            self.actual_agent_position=new_state
            return self.repetitive_penalty,new_state,False
    
        cell= self.structure[new_state[0]][new_state[1]]

        print(new_state,"Nuevo estado",cell,"celda")

        if cell=="G":
            self.structure[new_state[0]][new_state[1]]="A"
            self.structure[self.actual_agent_position[0]][self.actual_agent_position[1]]="."
            self.actual_agent_position=new_state
            return self.goal_reward,new_state,True
        elif cell=="#":
            return reward_obsta,new_state,False
        elif cell==".":
            self.structure[new_state[0]][new_state[1]]="A"
            self.visited_positions.add(tuple(new_state))
            self.structure[self.actual_agent_position[0]][self.actual_agent_position[1]]="."
            self.actual_agent_position=new_state
            return reward_pass,new_state,False
    
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

    def render(self):
        colors=["black","white","green","yellow"]
        cmap=ListedColormap(colors)

        patches=[
            Rectangle(xy=(0,0),width=1,height=1,facecolor=colors[0],label="Obstaculo"),
            Rectangle(xy=(0,0),width=1,height=1,facecolor=colors[1],label="Camino"),
            Rectangle(xy=(0,0),width=1,height=1,facecolor=colors[2],label="Agente"),
            Rectangle(xy=(0,0),width=1,height=1,facecolor=colors[3],label="Meta")]

        render_structure=copy.deepcopy(self.structure)

        for i in range(len(render_structure)):
            for j in range(len(render_structure[i])):
                match render_structure[i][j]:
                    case "#":
                        render_structure[i][j]=0
                    case ".":
                        render_structure[i][j]=1
                    case "A":
                        render_structure[i][j]=2
                    case "G":
                        render_structure[i][j]=3
        
        render_structure=np.array(render_structure,dtype=np.int32)
        
        plt.clf()
        plt.imshow(render_structure,cmap=cmap)
        plt.legend(handles=patches)
        plt.axis("off")
        plt.pause(0.2)
        plt.draw()

        plt.savefig("laberinto.png")

class Agent:

    def __init__(self,input):

        self.input=input

        self.model=tf.keras.Sequential()
        self.model.add(tf.keras.layers.Conv2D(32,(3,3),input_shape=self.input,activation="relu"))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(128,activation="relu"))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(4,activation="linear"))

        self.best_avg_reward=-float("inf")
        self.memory_maxlen=5000

        self.historial_rewards=[]

        self.gamma=0.99
        self.epsilon=1.0
        self.epsilon_min=0.1
        self.epsilon_decay=0.99
        self.learning_rate=0.005
        self.batch_size=16
        self.memory=deque(maxlen=5000)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mean_squared_error"
        )

    def update_memory_needed(self,episode,reward):

        self.historial_rewards.append(reward)

        if episode%50==0 and len(self.historial_rewards)>=50:
            last_rewards=self.historial_rewards[-50:]
            average=sum(last_rewards)/len(last_rewards)

            if average>self.best_avg_reward:
                self.best_avg_reward=average
            else:
                if self.memory_maxlen<11000:
                    self.memory_maxlen+=200

                if self.memory_maxlen> len(self.memory):
                    self.memory=deque(list(self.memory),maxlen=self.memory_maxlen)
    
    def get_action(self,state):

        random_number=tf.random.uniform(shape=[],dtype=tf.float32)
        print(random_number,"random epsilon")

        if random_number < self.epsilon:
            desicion=tf.random.uniform(dtype=tf.int32,minval=0,maxval=4,shape=[])
            print(desicion,"Decision")
            return desicion
        else:
            state=np.expand_dims(state,axis=0)
            prediction=self.model.predict(state)
            print(np.argmax(prediction[0]),"Prediccion Modelo")
            return np.argmax(prediction[0])
        
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))

    def replay(self):

        if len(self.memory)<self.batch_size:
            return

        random_batch=random.sample(self.memory,self.batch_size)

        for experience in random_batch:
            state,action,reward,next_state,done=experience

            state=np.expand_dims(state,axis=0)
            next_state=np.expand_dims(next_state,axis=0)

            q_values=self.model.predict(state)
            q_next=self.model.predict(next_state)

            if done:
                q_target=reward
            else:
                q_target=reward+self.gamma*max(q_next[0])

            q_values[0][action]=q_target
            self.model.fit(state,q_values,verbose=0)




        




