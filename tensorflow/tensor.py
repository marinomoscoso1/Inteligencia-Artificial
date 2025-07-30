#import tensorflow as tf 


def create_structure(file:str)->list:
    """Crea matriz en base a una estructura de un archivo 
       txt y confirma que todo este acorde
    """

    with open(file,"r") as f:
        read= f.readlines()

    structure=[]

    for i in read:
        i=i.replace("\n","").split()
        for c in i:
            if c not in ["A","#",".","G"]:
                raise "caracteres invalidos"
            
        structure.append(i)

    for f in structure:
        if len(f)!=len(structure[0]) and len(structure)==0:
            raise "Filas invalidas"

    is_A=False
    is_G=False

    agent_position=None
    goal_position=None

    for i in range(len(structure)):
        for j in range(len(structure[i])):
            if structure[i][j]=='A':
                is_A=True
                agent_position=[i,j]
            elif structure[i][j]=='G':
                is_G=True
                goal_position=[i,j]


    if is_A and is_G:
        return (structure,agent_position,goal_position)
    else:
        raise "matriz mal configurada"
    

def rewards(structure,state,action)->tuple[int,list,bool]:
    reward_goal=100
    reward_obsta=-10
    reward_pass=-1
    invalid_reward=-8

    match action:
        case 0:
            new_state=[state[0],state[1]-1]
        case 1:
            new_state=[state[0],state[1]+1]
        case 2:
            new_state=[state[0]+1,state[1]]
        case 3:
            new_state=[state[0]-1,state[1]]
    
    if new_state[0]<0 or new_state[1]<0 or new_state[0]>len(structure) or new_state[1]>=len(structure[0]):
        return invalid_reward,new_state,False
    
    cell= structure[new_state[0]][new_state[1]]

    if cell=="G":
        structure[new_state[0]][new_state[1]]="A"
        return reward_goal,new_state,True
    elif cell=="#":
        return reward_obsta,new_state,False
    elif cell==".":
        structure[new_state[0]][new_state[1]]="A"
        structure[state[0]][state[1]]="."
        for i in structure:
            print(i)
        return reward_pass,new_state,False

class Envirement:
    pass
        




