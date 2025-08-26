import numpy as np
import string

from gym import Env, spaces
from gym.utils import seeding
from gym import utils

import rendering.networkxMDPRendering as gRendering
import rendering.textMDPRendering as tRendering
import rendering.pydotMDPRendering as dRendering


def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


renderers = {'text': tRendering.textRenderer, 'pylab': gRendering.GraphRenderer, 'pydot':  dRendering.pydotRenderer}

class DiscreteMDP(Env):

    """
    Parameters
    - nS: number of states
    - nA: number of actions
    - P: transition distributions (*)
    - R: reward distributions (*)
    - isd: initial state distribution (**)

    (*) dictionary dict of dicts of lists, where
      P[s][a] == [(probability, nextstate, done), ...]
      R[s][a] == distribution(mean,param)
       One can sample R[s][a] using R[s][a].rvs()
    (**) list or array of length nS


    """

    #metadata = {'render.modes': ['text', 'ansi', 'pylab', 'pydot', 'maze']}


    def __init__(self, nS, nA, P, R, isd,nameActions=[],seed=None):
        self.nS = nS
        self.nA = nA
        self.P = P
        self.R = R

        self.isd = isd
        self.reward_range = (0, 1)

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)


        self.states = range(0,self.nS)
        self.actions = range(0,self.nA)


        # Rendering parameters and variables:
        self.lastaction=None
        self.lastreward=0.

        self.rendermode = ''
        self.initializedRenderer=False
        self.nameActions = nameActions
        if(len(nameActions)!=nA):
            self.nameActions = list(string.ascii_uppercase)[0:min(nA,26)]



        # Initialization
        self.seed(seed)
        self.reset()



    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction=None
        return self.s

    def step(self, a):
        """

        :param a: action
        :return:  (state, reward, IsDone?, meanreward)
        The meanreward is returned for information, it shold not begiven to the learner.
        """
        transitions = self.P[self.s][a]
        rewarddis = self.R[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, d= transitions[i]
        r =  rewarddis.rvs()
        m = rewarddis.mean()
        self.s = s
        self.lastaction=a
        self.lastreward=r
        return (s, r, d, m)

    def getTransition(self,s,a):
        transition = np.zeros(self.nS)
        for c in self.P[s][a]:
            transition[c[1]]=c[0]
        return transition
    

    def getMeanReward(self, s, a):
        rewarddis = self.R[s][a]
        r =  rewarddis.mean()
        return r




    def render(self, mode='pylab'):
        if (mode != ''):
            if ((not self.initializedRenderer) or  (self.rendermode != mode)):
                self.rendermode =  mode
                self.renderer = renderers[mode]()
                self.initializedRenderer = True
            self.renderer.render(self.states,self.actions,self.R,self.P,self.nameActions,self.s, self.lastaction,self.lastreward)




import scipy.stats as stat

class Dirac:
    def __init__(self,value):
        self.v = value
    def rvs(self):
        return self.v
    def mean(self):
        return self.v


class RandomMDP(DiscreteMDP):
    def __init__(self, nbStates,nbActions, maxProportionSupportTransition=0.5, maxProportionSupportReward=0.1, maxProportionSupportStart=0.2, minNonZeroProbability=0.2, minNonZeroReward=0.3, rewardStd=0.5, seed=None):
        self.nS = nbStates
        self.nA = nbActions
        self.states = range(0,self.nS)
        self.actions = range(0,self.nA)

        self.seed(seed)

        self.startdistribution = np.zeros((self.nS))
        self.rewards = {}
        self.transitions = {}
        self.P = {}
        # Initialize a randomly generated MDP
        for s in self.states:
            self.P[s]={}
            self.transitions[s]={}
            self.rewards[s]={}
            for a in self.actions:
                self.P[s][a]=[]
                self.transitions[s][a]={}
                my_mean = self.sparserand(p=maxProportionSupportReward, min=minNonZeroReward)
                if (rewardStd>0 and my_mean>0 and my_mean<1):
                    ma, mb = (0 - my_mean) / rewardStd, (1 - my_mean) / rewardStd
                    self.rewards[s][a] = stat.truncnorm(ma,mb,loc=my_mean,scale=rewardStd)
                else:
                    self.rewards[s][a] = Dirac(my_mean)
                transitionssa = np.zeros((self.nS))
                for s2 in self.states:
                    transitionssa[s2] = self.sparserand(p=maxProportionSupportTransition,min=minNonZeroProbability)
                mass = sum(transitionssa)
                if (mass > 0):
                    transitionssa = transitionssa / sum(transitionssa)
                    transitionssa = self.reshapeDistribution(transitionssa, minNonZeroProbability)
                else:
                    transitionssa[self.np_random.randint(self.nS)] = 1.
                li= self.P[s][a]
                [li.append((transitionssa[s], s, False)) for s in self.states if transitionssa[s]>0]
                self.transitions[s][a]= {ss:transitionssa[ss] for ss in self.states}

            self.startdistribution[s] = self.sparserand(p=maxProportionSupportStart,min=minNonZeroProbability)
        mass = sum(self.startdistribution)
        if (mass > 0):
            self.startdistribution = self.startdistribution / sum(self.startdistribution)
            self.startdistribution = self.reshapeDistribution(self.startdistribution, minNonZeroProbability)
        else:
            self.startdistribution[self.np_random.randint(self.nS)] = 1.

        checkRewards = sum([sum([self.rewards[s][a].mean() for a in self.actions]) for s in self.states])
        if(checkRewards==0):
            s = self.np_random.randint(0,self.nS)
            a = self.np_random.randint(0,self.nA)
            my_mean = minNonZeroReward + self.np_random.rand() * (1. - minNonZeroReward)
            if (rewardStd > 0 and my_mean > 0 and my_mean < 1):
                ma, mb = (0 - my_mean) / rewardStd, (1 - my_mean) / rewardStd
                self.rewards[s][a] = stat.truncnorm(ma, mb, loc=my_mean, scale=rewardStd)
            else:
                self.rewards[s][a] = Dirac(my_mean)
        #print("Random MDP is generated")
        #print("initial:",self.startdistribution)
        #print("rewards:",self.rewards)
        #print("transitions:",self.P)

        # Now that the Random MDP is generated with a given seed, we finalize its generation with an empty seed (seed=None) so that transitions/rewards are indeed stochastic:
        super(RandomMDP, self).__init__(self.nS, self.nA, self.P,  self.rewards, self.startdistribution,seed=None)

    def sparserand(self,p=0.5, min=0., max=1.):
        u = self.np_random.rand()
        if (u <= p):
            return min + self.np_random.rand() * (max - min)
        return 0.

    def reshapeDistribution(self,distribution, p):
        mdistribution = [0 if x < p else x for x in distribution]
        mass= sum(mdistribution)
        while(mass<0.99999999):
            i = self.np_random.randint(0, len(distribution))
            if(mdistribution[i]<p):
                newp = min(p, 1.-mass)
                if (newp==p):
                    mass = mass - mdistribution[i]+p
                    mdistribution[i]=p
            if (mdistribution[i] >= p):
                newp = min(1.,mdistribution[i] + 1.-mass)
                mass = mass- mdistribution[i]+newp
                mdistribution[i] = newp
        mass = sum(mdistribution)
        mdistribution = [x/mass for x in mdistribution]
        return mdistribution


class RiverSwim(DiscreteMDP):
    def __init__(self, nbStates, rightProbaright=0.6, rightProbaLeft=0.05, rewardL=0.1, rewardR=1.):#, ergodic=False): # TODO ergordic option
        self.nS = nbStates
        self.nA = 2
        self.states = range(0,self.nS)
        self.actions = range(0,self.nA)
        self.nameActions = ["R", "L"]


        self.startdistribution = np.zeros((self.nS))
        self.startdistribution[0] =1.
        self.rewards = {}
        self.P = {}
        self.transitions = {}
        # Initialize a RiverSwim MDP
        for s in self.states:
            self.P[s]={}
            self.transitions[s]={}
            # GOING RIGHT
            self.transitions[s][0]={}
            self.P[s][0]= [] #0=right", 1=left
            li = self.P[s][0]
            prr=0.
            if (s<self.nS-1):
                li.append((rightProbaright, s+1, False))
                self.transitions[s][0][s+1]=rightProbaright
                prr=rightProbaright
            prl = 0.
            if (s>0):
                li.append((rightProbaLeft, s-1, False))
                self.transitions[s][0][s-1]=rightProbaLeft
                prl=rightProbaLeft
            li.append((1.-prr-prl, s, False))
            self.transitions[s][0][s ] = 1.-prr-prl

            self.P[s][1] = []  # 0=right", 1=left
            self.transitions[s][1]={}
            li = self.P[s][1]
            if (s > 0):
                li.append((1., s - 1, False))
                self.transitions[s][1][s-1]=1.
            else:
                li.append((1., s, False))
                self.transitions[s][1][s]=1.

            self.rewards[s]={}
            if (s==self.nS-1):
                self.rewards[s][0] = Dirac(rewardR)
            else:
                self.rewards[s][0] = Dirac(0.)
            if (s==0):
                self.rewards[s][1] = Dirac(rewardL)
            else:
                self.rewards[s][1] = Dirac(0.)
                
        #print("Rewards : ", self.rewards, "\nTransitions : ", self.transitions)


        super(RiverSwim, self).__init__(self.nS, self.nA, self.P,  self.rewards, self.startdistribution,self.nameActions)
    
    
    
    
    
    
class ThreeState(DiscreteMDP):
    def __init__(self, delta = 0.005, fixed_reward = True):
        self.nS = 3
        self.nA = 2
        self.states = range(0,self.nS)
        self.actions = range(0,self.nA)
        self.nameActions = ["R", "L"]


        self.startdistribution = np.zeros((self.nS))
        self.startdistribution[0] =1.
        self.rewards = {}
        self.P = {}
        self.transitions = {}
        # Initialize a 3-state MDP

        s = 0
        self.P[s]={}
        self.transitions[s]={}
        # Action 0
        self.transitions[s][0]={}
        self.P[s][0]= [] #0=right", 1=left
        self.P[s][0].append((delta, 1, False))
        self.transitions[s][0][1] = delta
        self.P[s][0].append((1.- delta, 2, False))
        self.transitions[s][0][2] = 1. - delta
        # Action 1 is just the same for s = 0 and s = 1
        self.P[s][1] = []  # 0=right", 1=left
        self.transitions[s][1] = self.transitions[s][0]
        self.P[s][1] = self.P[s][0]
        # reward
        self.rewards[s]={}
        if fixed_reward:
            self.rewards[s][0] = Dirac(0.)
            self.rewards[s][1] = Dirac(0.)
        else:
            self.rewards[s][0] = Dirac(0.)
            self.rewards[s][1] = Dirac(0.)
        
        s = 1
        self.P[s]={}
        self.transitions[s]={}
        # Action 0
        self.transitions[s][0]={}
        self.P[s][0]= [] #0=right", 1=left
        self.P[s][0].append((1., 0, False))
        self.transitions[s][0][0] = 1.
        # Action 1 which just the same for s = 0 and s = 1
        self.P[s][1] = []  # 0=right", 1=left
        self.transitions[s][1] = self.transitions[s][0]
        self.P[s][1] = self.P[s][0]
        # reward
        self.rewards[s]={}
        if fixed_reward:
            self.rewards[s][0] = Dirac(1./3.)
            self.rewards[s][1] = Dirac(1./3.)
        else:
            self.rewards[s][0] = stat.bernoulli(1./3.)
            self.rewards[s][1] = stat.bernoulli(1./3.)
        
        s = 2
        self.P[s]={}
        self.transitions[s]={}
        # Action 0
        self.transitions[s][0]={}
        self.P[s][0]= [] #0=right", 1=left
        self.P[s][0].append((1., 2, False))
        self.transitions[s][0][2] = 1.
        # Action 1 which just the same for s = 0 and s = 1
        self.transitions[s][1]={}
        self.P[s][1]= [] #0=right", 1=left
        self.P[s][1].append((delta, 1, False))
        self.transitions[s][1][1] = delta
        self.P[s][1].append((1.- delta, 0, False))
        self.transitions[s][1][0] = 1. - delta
        # reward
        self.rewards[s]={}
        if fixed_reward:
            self.rewards[s][0] = Dirac(2./3.)
            self.rewards[s][1] = Dirac(2./3.)
        else:
            self.rewards[s][0] = stat.bernoulli(2./3.)
            self.rewards[s][1] = stat.bernoulli(2./3.)
         
        #print("Rewards : ", self.rewards, "\nTransitions : ", self.transitions)
        super(ThreeState, self).__init__(self.nS, self.nA, self.P,  self.rewards, self.startdistribution, self.nameActions)
        
        
