import random as rm

def create_structure(file:str="./structure.txt",random_maze:bool=False,rows:int=10,cols:int=10)->list:
    """Crea matriz en base a una estructura de un archivo 
       txt y confirma que todo este acorde
    """

    structure=[]

    is_A=False
    is_G=False

    agent_position=None
    goal_position=None

    if random_maze:
        valid_chars=[".","#","A","G"]

        structure=[[valid_chars[0] for _ in range(cols)] for _ in range(rows)]
        
        for i in range(len(structure)):
            for j in range(len(structure[i])):
                if i==0 or i==len(structure)-1:
                    structure[i][j]=valid_chars[1]
                elif j==0 or j==len(structure[i])-1:
                    structure[i][j]=valid_chars[1]
                elif rm.random()<0.3 and structure[i][j]==".":
                    structure[i][j]=valid_chars[1]

        while not is_A or not is_G:
            row=rm.randint(0,len(structure)-1)
            col=rm.randint(0,len(structure[0])-1)

            cell=structure[row][col]

            if cell=="." and not is_A:
                structure[row][col]="A"
                agent_position=[row,col]
                is_A=True
            elif cell=="." and not is_G and [row,col]!=agent_position:
                structure[row][col]="G"
                goal_position=[row,col]
                is_G=True
    else:
        with open(file,"r") as f:
            read= f.readlines()

        for i in read:
            i=i.replace("\n","").split()
            for c in i:
                if c not in ["A","#",".","G"]:
                    raise "caracteres invalidos"
            
            structure.append(i)

    for f in structure:
        if len(f)!=len(structure[0]):
            raise Exception("Filas invalidas")
        
    if agent_position is None and goal_position is None:
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
        raise Exception("matriz mal configurada")
    
create_structure(random_maze=True)
    