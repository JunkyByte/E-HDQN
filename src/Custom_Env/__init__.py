from gym.envs.registration import register

register(
    id='DimGrid-v0',
    entry_point='Custom_Env.DimensionalGrid:DimGridEnvironment',
    kwargs={'size': 10, 'hard': False},
)

register(
    id='DimGridHard-v0',
    entry_point='Custom_Env.DimensionalGrid:DimGridEnvironment',
    kwargs={'size': 10, 'hard': True},
)

