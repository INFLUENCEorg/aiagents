from gym.spaces.space import Space


class FixedActionSpace(Space):
  """
  FixedActionSpace assumes that there is a enumerable set 
  of all possible actions that can become possible in the environment,
  even if it would run infinitely long.
  """

  def getAllActions(self) -> dict:
      """
      get the complete dict of all actions that might become
      possible in this actionspace.
      the key is the int action number,
      the value a human-readable action value.
      eg {0:"UP", 1:"DOWN"}.
      """  
      raise NotImplementedError
