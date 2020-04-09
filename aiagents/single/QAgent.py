from aiagents.single.AtomicAgent import AtomicAgent
from aiagents.QAgentComponent import QAgentComponent
from aienvs.Environment import Env
from gym import spaces
from math import exp
import random
from aiagents.utils.Hashed import Hashed
from aienvs.gym.DecoratedSpace import DecoratedSpace
from gym.spaces import Dict

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

    chooseAction implements a standard epsilon-greedy
    """

    def __init__(
        self, 
        switchId:str, 
        actionspace:Dict=None, 
        observationspace=None, 
        parameters:dict={'alpha':0.1, 'gamma':1, 'epsilon': 0.1}
    ):
        super().__init__(switchId, actionspace, observationspace, parameters)
        # determine our action space, subset of env action_space
        self._lastAction = None
        self._lastState = None
        self._alpha = parameters['alpha']
        self._gamma = parameters['gamma']
        self._epsilon = parameters['epsilon']
        self._Q = {}  # Q[state][action]=Q value after _lastAction
        self._steps = 0
        self._eval = False # in eval mode, the agent executes the greedy policy given by the q function
        self._actionspace = DecoratedSpace.create(actionspace)
    
    def eval(self):
        self._eval = True

    def train(self):
        self._eval = False

    # Override
    def step(self, observation=None, reward:float=None, done:bool=None) -> dict:
        # observation is the current state
        newstate = Hashed(observation)
        # do not update q values in the first step or in evaluation mode
        if self._lastState != None and self._lastAction != None and self._eval == False:
            self._updateQ(self._lastState, self._lastAction, newstate, reward, done)

        action = None
        if done is True:
            self._lastState = None
        else:
            action = self._chooseAction(newstate)
            self._lastState = newstate
        self._lastAction = action
        self._steps = self._steps + 1
        return {self.agentId: action}

    # Override
    def getQ(self, state, action) -> float:
        return self._getQ(Hashed(state), action)

    # Override
    def getV(self, state):
        return None  # what should this do anyway?

    ################### PRIVATE ##############
    def _getQ(self, state:Hashed, action:int):
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

    def _updateQ(self, oldstate:Hashed, action:int, newstate:Hashed, reward:float, done:bool):
        """
        Updates our Q[state][act]->float dict
        according to the wiki formula.
        action was applied in old state, and we
        got in new state with a reward
        _lastState and _lastAction MUST be set properly.
        @param oldstate the old state, Hashed
        @param action, the action that brought us from old to new state. int
        @param newstate the new state, Hashed
        @param reward the reward associated with going from the old to the
        new state with action.
        """
        Qsa = self._getQ(oldstate, action)
        Qs1a = 0
        # if the next state is the terminal state, then the max q value of the next state is 0
        if done is False: 
            Qs1a = self._getMaxQ(newstate)
        Qnew = (1 - self._alpha) * Qsa + self._alpha * (reward + self._gamma * Qs1a)
        if not oldstate in self._Q.keys():
            self._Q[oldstate] = {}
            for a in range(self._actionspace.getSize()):
                self._Q[oldstate][a] = INITIAL_Q
        self._Q[oldstate][action] = Qnew

    def _getMaxQ(self, state:Hashed):
        """
        @param the state, Hashed
        @return maximum possible Q(state, action) for any action, or INITIAL_Q
        if state does not have any Q value.
        """
        if not state in self._Q.keys():
            return INITIAL_Q
        maxQ = max(self._Q[state].values())
        return maxQ

    def _getMaxAction(self, state:Hashed) -> int:
        """
        @param the state, Hashed
        @return the action (int) that has the maximum possible Q(state, action),
        or None if there is no Q(state,action)
        """
        if not state in self._Q:
            return None

        Qs = self._Q[state]

        maxQ = float('-inf')
        maxAction = None

        for act in Qs.keys():
            if Qs[act] > maxQ:
                maxQ = Qs[act]
                maxAction = act
        return maxAction

    def _chooseAction(self, state:Hashed) -> int:
        """
        @param state the Hashed state
        The Q agent chooses either
        (A) an arbitrary action
        (B) the best action, that one which currently has highest Q

        @return our next action, int (the index in the action_space which is a decoratedspace)
        """
        bestact = None
        if self._eval == True or random.uniform(0, 1) >= self._epsilon:
            bestact = self._getMaxAction(state)

        if bestact == None:
            # sample: Dict -> OrderedDict
            # bestact = something like self._actionspace.sample()
            bestact = random.randint(0, self._actionspace.getSize() - 1)

        return bestact
