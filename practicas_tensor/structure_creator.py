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
    