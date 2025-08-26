from runExperiments import *


def demo_riverSwim():
     testName = 'riverSwim6'
     envName = (bW.registerWorlds[testName])(0)
     env = bW.makeWorld(envName)
     learner = lr.Random(env)
     #learner = lh.Human(env)
     # learner = le.UCRL3_lazy(env.observation_space.n, env.action_space.n, delta=0.05)
     animate(env, learner, 100, 'pylab')
     #animate(env, learner, 100, 'text')

def demo_randomGrid():
     testName = 'random_grid'
     envName = (bW.registerWorlds[testName])(0)
     env = bW.makeWorld(envName)
     learner = lr.Random(env)
     # learner = le.UCRL3_lazy(env.observation_space.n, env.action_space.n, delta=0.05)
     animate(env, learner, 100, 'maze')

def demo_randomMDP():
    testName = 'random10'
    envName = (bW.registerWorlds[testName])(0)
    env = bW.makeWorld(envName)
    learner = lr.Random(env)
    # learner = le.UCRL3_lazy(env.observation_space.n, env.action_space.n, delta=0.05)
    animate(env, learner, 50, 'pylab')
    #animate(env, learner, 50, 'text')
    #
    #
    # testName = 'random100'
    # envName = (bW.registerWorlds[testName])(0)
    # env = bW.makeWorld(envName)
    # learner = lr.Random(env)
    # # learner = le.UCRL3_lazy(env.observation_space.n, env.action_space.n, delta=0.05)
    # animate(env, learner, 100, 'text')

#demo_randomMDP()

demo_riverSwim()