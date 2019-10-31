from aiagents.single.AtomicAgent import AtomicAgent
from aiagents.QAgentComponent import QAgentComponent
from aienvs.Environment import Env
from gym import spaces
from math import exp
import random
from builtins import None

INITIAL_Q = 0


class QAgent(AtomicAgent, QAgentComponent):
    """
    Agent that learns Q[state][action]->float the quality of each
    action in each state from the rewards that are received
    from the environment. Notice that the reward is received only
    then next time step is called, therefore the Q lags behind one step.
    see https://en.wikipedia.org/wiki/Q-learning.
    
    The Q learning algorithm has two parameters:
    * alpha in [0,1] the learning rate. Bigger means that new rewards
        are having more effect on the Q, so that learning goes
        faster but becomes less stable
    * gamma in [0,1], the discount factor, determines how much
       of the new state's Q is incorporated into the old state's Q
       and thus determines how quick future effects boil down
       to the current Q.

    see also chooseAction which has  2 parameters
    * m: the midpoint of the sigmoid. This determines where the 
        behavour switches from A to B.
    * s: the steepness of the cross-over from A to B behaviour. values closer
        to 1 give a slower change from A to B.
    """

    def __init__(self, switchId:str, environment:Env,
                 parameters:dict={'alpha':0.1, 'gamma':1, 'm':500, 's':0.01}):
        super(AtomicAgent, self).__init__(switchId, environment, parameters)
        self._lastAction = None
        self._lastState = None
        self._alpha = parameters['alpha']
        self._gamma = parameters['gamma']
        self._m = parameters['m']
        self._s = parameters['s']
        self._Q = {}  # Q[state][action]=Q value after _lastAction
        self._steps = 0

    #Override
    def step(self, observation=None, reward:float=None, done:bool=None) -> spaces.Dict:
        # observation is the current state
        newstate = Hashed(observation)
        if self._lastAction != None:
            self._updateQ(self._lastState, self._lastAction, newstate, reward)
        
        action = self._chooseAction(newstate)
        self._lastState = newstate
        self._lastAction = action
        self._steps = self._steps + 1
        return spaces.Dict({self.agentId(), action})

    #Override
    def getQ(self, state, action) -> float:
        return self._getQ(Hashed(state),Hashed(action))
    
    def _getQ(self,state:Hashed, action:Hashed):
        """
        @param state the state, not Hashed
        @param action the action, not Hashed
        @return the current Q value for this state and action, or
        INITIAL_Q if no Q stored for this state,action
        """
        if state in self._Q.keys():
            if action in self._Q[state].keys():
                return self._Q[state][action]
        return INITIAL_Q

    
    def _updateQ(self, oldstate:Hashed, action:Hashed, newstate:Hashed, reward:float):
        """
        Updates our Q[state][act]->float dict 
        according to the wiki formula.
        action was applied in old state, and we 
        got in new state with a reward
        _lastState and _lastAction MUST be set properly.
        @param oldstate the old state, Hashed
        @param action, the action that brought us from old to new state. Hashed
        @param newstate the new state, Hashed
        @param reward the reward associated with going from the old to the 
        new state with action.
        """
        Qsa = self.getQ(oldstate, action)
        Qs1a = self._getmaxQ(newstate)
        Qnew = (1 - self._alpha) * Qsa + self._alpha * (reward + self._gamma * Qs1a)
        self._Q[oldstate][action] = Qnew
            
    def _getMaxQ(self, state):
        """
        @param the state, hashable
        @return maximum possible Q(state, action) for any action, or INITIAL_Q 
        if state does not have any Q value.
        """
        if not state in self.Q.keys():
            return INITIAL_Q
        return max(self.Q[state].values())
    
    def _getMaxAction(self,state:Hashed) -> Hashed:
        """
        @param the state, Hashed
        @return the Hashed action  that has the maximum possible Q(state, action),
        or None if there is no Q(state,action)
        """
        if not state in self._Q:
            return None
        
        Qs = self._Q[state]
        maxQ = float('-inf')
        maxAction=None
        
        for act in Qs.keys()
            if Qs[act] > maxQ:
                maxQ=Qs[act]
                maxAction=act
        return maxAction
    
        
    def _chooseAction(self, state:Hashed) -> Hashed:
        """
        @param state the Hashed state
        The Q agent chooses either 
        (A) an arbitrary action 
        (B) the best action, that one which currently has highest Q
    
        The choice between A and B is made by the function p(N) (see below).
        p(N) is the chance that B is chosen (so 1-p is the chance for A)
        It has two parameters:
        * m: This determines where the behavour switches from A to B or vice versa.
        * s: how quick the behaviour changes.
        If positive, behaviour changes gradually from B to A.
        If negative, behaviour changes gradually from A to B.
        the bigger the absolute value, the faster the change.
        Usually the value is close to 0 to make slow changes.
        If 0, behaviour is constant everywhere, determined only by m

        If strategy B was picked but Q is empty, we revert to strategy A.
        @return our next action, Hashed 
        """
        bestact=None
        if random.uniform(0,1) <= self._p():
            bestact = self._getMaxAction(state)
        
        if bestact==None:
            bestact=Hashed(self._environment.action_space.sample())

        return bestact
        
    def _p(self):
        """
        p(N) = 1/(1+exp(s(m - N)) a adjustable sigmoid function
        
        p(N) has two parameters:
        * m: the midpoint of the sigmoid. 
        * s: the steepness of the cross-over from A to B behaviour.
        N is the number of steps taken so far.
        """
        return 1 / (1 + exp(self._s * (self._m - self._steps)))

class Hashed:
    """
    Class that makes any object hashable by encapsulation.
    Assumes the object is not changed.
    If it is not immutable, and it is mutated,
    then the hashcode will become wrong.
    """
    def __init__(self,obj):
        self._obj = obj
        try:
            self._hash = hash(obj)
        except TypeError:
            self._hash = hash(str(obj))
            
    def get(self):
        """
        @return the original object
        """
        return self._obj
    
    def __eq__(self,other):
        return isinstance(other, Hashed) and self._obj==other._obj
    
    def __hash__(self):
        return self._hash
        
        
        