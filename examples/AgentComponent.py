from aiagents.multi.ComplexAgentComponent import ComplexAgentComponent
from aiagents.single.RandomAgent import RandomAgent

def main():

    N_agents=10
    i=0
    scope=[1,2,3]
    simpleComponentList=[]
    verbose=True

    while( i < N_agents ):
        simpleComponentList.append(RandomAgent(i, scope, verbose))
        i+=1

    myComplexComponent=ComplexAgentComponent(simpleComponentList, verbose)

    N_steps=3
    i=0
    state=0

    while i<N_steps:
        myComplexComponent.observe(state)
        myComplexComponent.select_actions()
        i+=1
    return 0
  
if __name__== "__main__":
  main()

