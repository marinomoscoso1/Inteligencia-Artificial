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
        structure.append(i)

    is_A=False
    is_G=False

    agent_position=None
    goal_position=None

    for i in structure:
        for j in i:
            if structure[i][j]=='A':
                is_A=True
                agent_position=[i,j]
            elif structure[i][j]=='G':
                is_G=True
                goal_position=[i,j]


    if is_A and is_G:
        return (structure,agent_position,goal_position) if structure else None
    

def rewards():
    reward_goal=100
    reward_obsta=-10
    reward_pass=-1
    invalid_reward=-8




