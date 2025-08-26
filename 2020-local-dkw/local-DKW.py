######################################################################################################
#  Odalric-Ambrym Maillard
#  Last update: 04-2021
######################################################################################################

from cycler import cycler
import math
import scipy.special as sp
import numpy as np
import pylab as pl
import os


import pickle
from pynverse import inversefunc

######################################################################################################
#  Summary of principal functions
# proba_left(epsilon, n, a, b), proba_log_left(epsilon, n, a, b, log_threshold=1e-300)
# proba_right(epsilon, n, a, b), proba_log_right(epsilon, n, a, b, log_threshold=1e-300)
# epsilon_left(a, b, delta, n, epsilon0=0., epsilon1=1.)
# epsilon_right(a, b, delta, n, epsilon0=0., epsilon1=1.)
# DKW(delta,n)
#
#  See demo() for plots.
######################################################################################################


######################################################################################################
# Setup for color-blind printing.
######################################################################################################
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
default_cycler = (cycler(color=CB_color_cycle) +  cycler(marker=['o', '+', '^', 'd', 's', 'x','*','p','>'])+ cycler(linestyle=['-', '--', '-.', ':','-','--','-.',':','-']))
default_cycler = (cycler(color=CB_color_cycle) + cycler(marker=['o', '^', 'd', 's', '*', '>', 'x','p','+']))
#markeverycases = [None,         8,         (30, 8),         [16, 24, 30], [0, -1],         slice(100, 200, 3),         0.1, 0.3, 1.5,         (0.0, 0.1), (0.45, 0.1)]
#markeverycase= (0.0,0.1)#[0, -1]#slice(100, 200, 3)#(0.0,0.1)
pl.rc('lines', linewidth=3,markersize=8)
pl.rc('axes', prop_cycle=default_cycler)
######################################################################################################


def strvec(x):
    s=""
    for i in x:
        s+=", "+str(i)
    return "["+s[2:]+"]"

#Useful for Arxiv file naming conventions.
def renamefiles():
    listdir=os.listdir(path='.')
    replace = {',':'-', ' ':'', '[':'', ']':'', '.':''}
    filelistnames = open('renaminglist.txt', 'w')
    for f in listdir:
        l = len(f)
        if (f[l-3:l]=='pdf'):
            #print(f[0:l-4], l, f[l - 3:l])
            s = f[0:l-4]
            ss= ""
            for c in s:
                if (c in replace.keys()):
                    ss+=replace[c]
                else:
                    ss+=c
            print(s, 'renamed in ', ss)
            print(ss, file=filelistnames)
            os.rename(r'' + f, r'' + ss + '.pdf')


######################################################################################################
# Illustrative plot of U_n-U and U-U_n
######################################################################################################
def illustrate_left_right():
    n=15
    data = np.random.rand(n)
    f = lambda u: (len(list(filter(lambda x: x <= u, data))) + 0.) / (n + 0.) -u

    X = range(0, 1000)
    X = [x / 1000 for x in X]

    pl.clf()
    pl.plot(X, [f(x) for x in X],markevery=markeverycase)

    #pl.title('eps -> Pr( sup U_n-U >= eps)')
    #pl.ylim(bottom=0,top=1.)
    # pl.ylim(bottom=10e-10)
    # pl.xscale('log')
    #pl.yscale('log')
    #pl.legend()
    pl.savefig('Illustrate_left.png')
    pl.savefig('Illustrate_left.pdf')
    pl.savefig('Illustrate_left.ps')

    n=15
    data = np.random.rand(n)
    f = lambda u: u-(len(list(filter(lambda x: x <= u, data))) + 0.) / (n + 0.)

    X = range(0, 1000)
    X = [x / 1000 for x in X]

    pl.clf()
    pl.plot(X, [f(x) for x in X],markevery=markeverycase)

    #pl.title('eps -> Pr( sup U_n-U >= eps)')
    #pl.ylim(bottom=0,top=1.)
    # pl.ylim(bottom=10e-10)
    # pl.xscale('log')
    #pl.yscale('log')
    #pl.legend()
    pl.savefig('Illustrate_right.png')
    pl.savefig('Illustrate_right.pdf')
    pl.savefig('Illustrate_right.ps')

######################################################################################################
#  Auxiliary search functions
#
######################################################################################################

def finv_decreasing(f,y,down,up,epsilon=0.0001):
    mid = (up + down) / 2
    if (up - down > epsilon):
        if (f(mid)<= y ):
            return  finv_decreasing(f, y, down, mid)
        else:
            return  finv_decreasing(f, y, mid, up)
    else:
        if (f(mid)<= y ):
            return mid
        else:
            return up


def search_down(f, up, down, epsilon=0.0001):
    mid = (up + down) / 2
    if (up - down > epsilon):
        if f(mid):
            return search_down(f, mid, down)
        else:
            return search_down(f, up, mid)
    else:
        if f(down):
            return down
        return up


######################################################################################################
#  Computation of Probability functions
#
######################################################################################################
# sup U_n-U
def proba_left(epsilon, n, a, b):
    na = (int) (np.ceil(n * (1 - a - epsilon)))
    nb = n * (1 - b - epsilon)
    nb_ = (int) (np.floor(n * (1 - b - epsilon)))
    m = min(nb_+1,na-1)
    if na >= 0:
        if nb_ > 0:
            term1 = 0
            for ll in range(0,m+1):
                term1 += sp.binom(n, ll) * (min(1 - ll / n - epsilon, b) ** (n - ll)) * (1 - b) ** ll
            term2 = 0
            for ll in range(m+1, na):
                term2a = sp.binom(n, ll) * ((1 - ll / n - epsilon) ** (n - ll))
                term2b = epsilon*  (ll / n + epsilon) * (ll- 1)
                for j in range(0,m):
                    term2b += ((nb - j) / n) * sp.binom(ll, j) * ((ll - nb) / n) ** (ll - j - 1) * (1 - b) ** j
                term2 += term2a * term2b
            proba = term1 +term2
            #print(f'n, kb > 0, {proba, term1, term2}')
        else:
            term = 0
            for ll in range(0,na):
                term +=  sp.binom(n, ll)  *((1 - ll / n - epsilon) ** (n - ll)) * epsilon * (
                        ll / n + epsilon) ** (ll - 1)
            proba = term
            #print(f'n, kb < 0, {proba, term}')
    else:
        proba =0.
    print(f'{proba, epsilon,n,a,b}')
    return proba

# For accurate computations, using log.
def proba_log_left(epsilon, n, a, b, log_threshold=1e-300):
    if epsilon<= 0:
        return 1.
    if epsilon >= (1 - a):
        return 0.

    na_ceil = int(math.ceil(n * (1 - a - epsilon)))
    nb = n * (1 - b - epsilon)
    m = min(math.floor(nb)+1, na_ceil - 1)

    if nb > 0:
        term1 = 0
        for l in range(max(m + 1, 0)):
            z1 = 1 - l / n - epsilon
            if z1 > log_threshold:
                z2 = math.exp((n - l) * math.log(min(z1, b)))
                if z2 > log_threshold:
                    term1_log = math.log(sp.comb(n, l, exact=True)) + \
                                math.log(z2) + \
                                l * math.log(1 - b)
                    term1 += math.exp(term1_log)
        term2 = 0
        for l in range(max(m + 1, 0), max(na_ceil, 0)):
            z1 = 1 - l / n - epsilon
            if z1 > log_threshold:
                z2 = math.exp((n - l) * math.log(z1))
                if z2 > log_threshold:
                    term2a_log = math.log(sp.comb(n, l, exact=True)) + \
                                 math.log(z2)
                    term2b_log = math.log(epsilon) + (l - 1) * math.log(l / n + epsilon)
                    term2b = math.exp(term2b_log)
                    for j in range(max(m, 0)):
                        if (j<nb):
                            term2b_log = math.log(nb - j) - math.log(n) + math.log(sp.comb(l, j, exact=True)) + \
                                     (l - j - 1) * (math.log(l - nb) - math.log(n)) + j * math.log(1 - b)
                            term2b += math.exp(term2b_log)
                    term2_log = term2a_log + math.log(term2b)
                    term2 += math.exp(term2_log)
        proba = term1 + term2

        #print(f'kb > 0, {n, proba, term1, term2}')
    else:
        term = 0
        for l in range(max(na_ceil, 0)):
            z1 = (1 - l / n - epsilon)
            if z1 > log_threshold:
                z2 = math.exp((n - l) * math.log(z1))
                if z2 > log_threshold:
                    term_log = math.log(sp.comb(n, l, exact=True)) + \
                               math.log(z2) \
                               + math.log(epsilon) + (l - 1) * math.log((l / n + epsilon))
                    term += math.exp(term_log)
        proba = term

        #print(f'kb < 0, {n, proba}')
    return proba


# sup U-U_n
def proba_right(epsilon, n, a, b):
    return proba_left(epsilon,n,1-b,1-a)

# For accurate computations.
def proba_log_right(epsilon, n, a, b, log_threshold=1e-300):
    return proba_log_left(epsilon,n,1-b,1-a, log_threshold)




######################################################################################################
#  Plot probability functions
#
######################################################################################################
def plot_proba_left(ns):
    a=0.
    bs = [0.05,0.1,0.2,0.5,0.9,1.]
    pl.clf()
    X = range(0,1000)
    X = [x/1000 for x in X]
    markeverycase= (0.0,0.1)

    for n in ns:
        for b in bs:
            pl.plot(X,[proba_log_left(x, n, a, b) for x in X],label='['+str(a)+','+str(b)+'], n='+str(n),markevery=markeverycase)
            #epsilon(a, b, 0.05, n, epsilon0=0., epsilon1=1.)


    pl.title('eps -> Pr( sup U_n-U > eps)')
    pl.ylim(bottom=10e-10)
    #pl.ylim(bottom=10e-10)
    #pl.xscale('log')
    pl.yscale('log')
    pl.legend()
    pl.savefig('delta_'+str(a)+'_'+strvec(bs)+'_n'+strvec(ns)+'.png')
    pl.savefig('delta_'+str(a)+'_'+strvec(bs)+'_n'+strvec(ns)+'.pdf')
    pl.savefig('delta_'+str(a)+'_'+strvec(bs)+'_n'+strvec(ns)+'.ps')

    b=1.
    a_s = [0.0,0.1,0.5,0.8,0.9,0.95]
    pl.clf()
    X = range(0,1000)
    X = [x/1000 for x in X]

    for n in ns:
        for a in a_s:
            pl.plot(X,[proba_log_left(x, n, a, b) for x in X],label='['+str(a)+','+str(b)+'], n='+str(n),markevery=markeverycase)
            #epsilon(a, b, 0.05, n, epsilon0=0., epsilon1=1.)


    pl.title('eps -> Pr( sup U_n-U > eps)')
    pl.ylim(bottom=10e-10)
    #pl.ylim(bottom=10e-10)
    #pl.xscale('log')
    pl.yscale('log')
    pl.legend()
    pl.savefig('delta_'+strvec(a_s)+'_'+str(b)+'_n'+strvec(ns)+'.png')
    pl.savefig('delta_'+strvec(a_s)+'_'+str(b)+'_n'+strvec(ns)+'.pdf')
    pl.savefig('delta_'+strvec(a_s)+'_'+str(b)+'_n'+strvec(ns)+'.ps')

def plot_proba_right(ns):
    b=1.
    a_s = [0.0,0.1,0.5,0.8,0.9,0.95]
    pl.clf()
    X = range(0,1000)
    X = [x/1000 for x in X]
    markeverycase = (0.0, 0.1)

    for n in ns:
        for a in a_s:
            pl.plot(X,[proba_log_right(x, n, a, b) for x in X],label='['+str(a)+','+str(b)+'], n='+str(n),markevery=markeverycase)
            #epsilon(a, b, 0.05, n, epsilon0=0., epsilon1=1.)


    pl.title('eps -> Pr( sup U-U_n > eps)')
    pl.ylim(bottom=10e-10)
    #pl.ylim(bottom=10e-10)
    #pl.xscale('log')
    pl.yscale('log')
    pl.legend()
    pl.savefig('tildedelta_'+strvec(a_s)+'_'+str(b)+'_n'+strvec(ns)+'.png')
    pl.savefig('tildedelta_'+strvec(a_s)+'_'+str(b)+'_n'+strvec(ns)+'.pdf')
    pl.savefig('tildedelta_'+strvec(a_s)+'_'+str(b)+'_n'+strvec(ns)+'.ps')

    a=0.
    b_s = [0.05,0.1,0.2,0.5,0.9,1.]
    pl.clf()
    X = range(0,1000)
    X = [x/1000 for x in X]

    for n in ns:
        for b in b_s:
            pl.plot(X,[proba_log_right(x, n, a, b) for x in X],label='['+str(a)+','+str(b)+'], n='+str(n),markevery=markeverycase)
            #epsilon(a, b, 0.05, n, epsilon0=0., epsilon1=1.)


    pl.title('eps -> Pr( sup U-U_n > eps)')
    pl.ylim(bottom=10e-10)
    #pl.ylim(bottom=10e-10)
    #pl.xscale('log')
    pl.yscale('log')
    pl.legend()
    pl.savefig('tildedelta_'+str(a)+'_'+strvec(b_s)+'_n'+strvec(ns)+'.png')
    pl.savefig('tildedelta_'+str(a)+'_'+strvec(b_s)+'_n'+strvec(ns)+'.pdf')
    pl.savefig('tildedelta_'+str(a)+'_'+strvec(b_s)+'_n'+strvec(ns)+'.ps')



######################################################################################################
#  Computation of Inverse probability functions
#
######################################################################################################

def epsilon_left(a, b, delta, n, epsilon0=0., epsilon1=1.):
    #if (n == 0):
    #    return 1.
    #i=inversefunc(lambda x: proba_log_left(x, n, a, b), y_values=delta,domain=[epsilon0,epsilon1], open_domain=[True,False], image= [0.,1.])
    #print('[' + str(a) + ',' + str(b) + '], n=' + str(n), i)

    if (n == 0):
        return 1.
    i= finv_decreasing(lambda x: proba_log_left(x, n, a, b), delta, epsilon0, epsilon1, epsilon=0.0000001)
    print('[' + str(a) + ',' + str(b) + '], n=' + str(n), i, 'check:', proba_log_left(i, n, a, b), ':', delta)


    #i= search_down(lambda x: proba_log_left(x, n, a, b) <= delta , epsilon1, epsilon0, epsilon=0.001)
    #print('[' + str(a) + ',' + str(b) + '], n=' + str(n), i)

    return i

def epsilon_right(a,b,delta,n,epsilon0=0., epsilon1=1.):
    #if (n == 0):
    #    return 1.
    #i=inversefunc(lambda x: proba_log_left(x, n, a, b), y_values=delta,domain=[epsilon0,epsilon1], open_domain=[True,False], image= [0.,1.])
    #print('[' + str(a) + ',' + str(b) + '], n=' + str(n), i)

    if (n == 0):
        return 1.
    i= finv_decreasing(lambda x: proba_log_right(x, n, a, b), delta, epsilon0, epsilon1, epsilon=0.0000001)
    print('[' + str(a) + ',' + str(b) + '], n=' + str(n), i)
    #i= search_down(lambda x: proba_log_left(x, n, a, b) <= delta , epsilon1, epsilon0, epsilon=0.001)
    #print('[' + str(a) + ',' + str(b) + '], n=' + str(n), i)

    return i


def DKW(delta,n):
    if delta>0 and n>0:
        return min(np.sqrt(np.log(1 / delta) / (2 * n)), 1.)
    else:
        if (delta==0):
            return None
        else:
                return 1.



#  Computation of inverse probability as a function of n, plus storage in .data file
def compute_epsilon_left(a, b, delta):
    N = [2,3,4,5,6,8,10,12,14,16,19,22,25,28,32,36,40,45,50,55,62,69,76,87,96,108,120,132,147,162,180,198,208,230,252,265,288,315,345,380,420,460,500]
    Y = np.zeros(len(N))
    epsilon_t=1.
    epsilon_t_ = 0.
    for i in range(len(N)):
        e = epsilon_left(a, b, delta, N[i], epsilon0=epsilon_t_, epsilon1=epsilon_t)
        Y[i] = e
        print(N[i],Y[i])
        if (i>2):
            epsilon_t=min(e + 5*np.abs(Y[i-2]-Y[i-1])+10e-5,1.)
            if (N[i]>100):
                epsilon_t_=max(e-5*np.abs(Y[i-2]-Y[i-1])-10e-5,0.)
            #print(epsilon_t_,epsilon_t)

    filename = "epsilon_"+str(a)+'_'+str(b)+'_'+str(delta)+'.data'
    file =  open(filename,'wb')
    pickle.dump(Y, file)
    file.close()

def compute_epsilon_right(a, b, delta):
    N = [2,3,4,5,6,8,10,12,14,16,19,22,25,28,32,36,40,45,50,55,62,69,76,87,96,108,120,132,147,162,180,198,208,230,252,265,288,315,345,380,420,460,500]
    Y = np.zeros(len(N))
    epsilon_t=1.
    epsilon_t_ = 0.
    for i in range(len(N)):
        e = epsilon_right(a, b, delta, N[i], epsilon0=epsilon_t_, epsilon1=epsilon_t)
        Y[i] = e
        print(N[i],Y[i])
        if (i>2):
            epsilon_t=min(e + 5*np.abs(Y[i-2]-Y[i-1])+10e-5,1.)
            if (N[i]>100):
                epsilon_t_=max(e-5*np.abs(Y[i-2]-Y[i-1])-10e-5,0.)
            #print(epsilon_t_,epsilon_t)

    filename = "tildeepsilon_"+str(a)+'_'+str(b)+'_'+str(delta)+'.data'
    file =  open(filename,'wb')
    pickle.dump(Y, file)
    file.close()

def load_epsilon_left(a, b, delta):
    filename = "epsilon_"+str(a)+'_'+str(b)+'_'+str(delta)+'.data'
    file_oracle = open(filename, 'rb')
    data = pickle.load(file_oracle)
    Y = data
    return Y

def load_epsilon_right(a, b, delta):
    filename = "tildeepsilon_"+str(a)+'_'+str(b)+'_'+str(delta)+'.data'
    file_oracle = open(filename, 'rb')
    data = pickle.load(file_oracle)
    Y = data
    return Y


def compute_epsilons(delta):
    a=0.
    for b in [0.05,0.1,0.2, 0.5, 0.9, 1.]:
        print("b=",b)
        compute_epsilon_left(a, b, delta)
        compute_epsilon_right(a, b, delta)
    b=1.
    for a in [0.0,0.1,0.5,0.8,0.9,0.95]:
        print("a=",a)
        compute_epsilon_left(a, b, delta)
        compute_epsilon_right(a, b, delta)

######################################################################################################
#  Plot and display of inverse probability function
#
######################################################################################################
def print_tabulated_epsilon(filename):
    file = open(filename, 'w')
    delta_s = [0.01, 0.02, 0.05]
    a_s = [0.0,0.1,0.5,0.8,0.9,0.95]
    b_s = [0.05,0.1,0.2, 0.5, 0.9, 1.]
    n_s = [2,3,4,5,6,8,10,12,14,16,19,22,25,28,32,36,40,45,50,55,62,69,76,87,96,108,120,132,147,162,180,198,208,230,252,265,288,315,345,380,420,460,500]

    for delta in delta_s:
        print('delta=',delta,file=file)
        s = 'n\t'
        for a in a_s:
            s += str(a) + ':' + str(1.) + '\t'
        for b in b_s:
            s += str(0.) + ':' + str(b) + '\t'
        print(s,file=file)
        for n in n_s:
            s=str(n)+'\t'
            for a in a_s:
                    s+= str(epsilon_left(a, 1., delta, n, epsilon0=0., epsilon1=1.))+'\t'
            for b in b_s:
                    s+= str(epsilon_left(0., b, delta, n, epsilon0=0., epsilon1=1.))+'\t'
            print(s,file=file)


# As a function of delta
def plot_epsilon_left(ns):
    a=0.
    b_s = [0.05,0.1,0.2,0.5,0.9,1.]

    pl.clf()
    X = range(0,1000)
    X = [x/1000 for x in X]
    markeverycase  = [(int) (10**(x/3)) for x in range(0,3*3)]

    for n in ns:
        for b in b_s:
            Y= [epsilon_left(a, b, x, n, epsilon0=0., epsilon1=1.) for x in X]
            pl.plot(X, Y, label='[' + str(a) + ',' + str(b) + '], n=' + str(n),markevery=markeverycase)
            #print('n=', n, '\t', a, ':', b, '\t')
            #[print(X[i], Y[i]) for i in range(len(X))]

    pl.plot(X,[DKW(x,n) for x in X],label='DKW',markevery=markeverycase)


    pl.title('delta -> epsilon_{[a,b]}(n,delta)')
    #pl.ylim(bottom=10e-40)
    #pl.ylim(bottom=10e-10)
    pl.xscale('log')
    pl.xlim(left=10e-4)
    #pl.yscale('log')
    pl.legend()
    pl.savefig('epsilon_'+str(a)+'_'+strvec(b_s)+'_n'+strvec(ns)+'.png')
    pl.savefig('epsilon_'+str(a)+'_'+strvec(b_s)+'_n'+strvec(ns)+'.pdf')
    pl.savefig('epsilon_'+str(a)+'_'+strvec(b_s)+'_n'+strvec(ns)+'.ps')

    b=1.
    a_s = [0.0,0.1,0.5,0.8,0.9,0.95]

    pl.clf()
    X = range(0,1000)
    X = [x/1000 for x in X]
    markeverycase  = [(int) (10**(x/3)) for x in range(0,3*3)]

    for n in ns:
        for a in a_s:
            pl.plot(X, [epsilon_left(a, b, x, n, epsilon0=0., epsilon1=1.) for x in X], label='[' + str(a) + ',' + str(b) + '], n=' + str(n),markevery=markeverycase)

    pl.plot(X,[DKW(x,n) for x in X],label='DKW',markevery=markeverycase)


    pl.title('delta -> epsilon_{[a,b]}(n,delta)')
    #pl.ylim(bottom=10e-40)
    #pl.ylim(bottom=10e-10)
    pl.xscale('log')
    pl.xlim(left=10e-4)
    #pl.yscale('log')
    pl.legend()
    pl.savefig('epsilon_'+strvec(a_s)+'_'+str(b)+'_n'+strvec(ns)+'.png')
    pl.savefig('epsilon_'+strvec(a_s)+'_'+str(b)+'_n'+strvec(ns)+'.pdf')
    pl.savefig('epsilon_'+strvec(a_s)+'_'+str(b)+'_n'+strvec(ns)+'.ps')


# As a function of delta
def plot_epsilon_right(ns):
    a=0.
    b_s = [0.05,0.1,0.2,0.5,0.9,1.]

    pl.clf()
    X = range(0,1000)
    X = [x/1000 for x in X]
    markeverycase  = [(int) (10**(x/3)) for x in range(0,3*3)]

    for n in ns:
        for b in b_s:
            Y = [epsilon_right(a, b, x, n, epsilon0=0., epsilon1=1.) for x in X]
            pl.plot(X, Y, label='[' + str(a) + ',' + str(b) + '], n=' + str(n),markevery=markeverycase)
            # print('n=',n,'\t',a,':',b,'\t')
            # [print(X[i],Y[i]) for i in range(len(X))]


    pl.plot(X,[DKW(x,n) for x in X],label='DKW',markevery=markeverycase)


    pl.title('delta -> epsilon_{[a,b]}(n,delta)')
    #pl.ylim(bottom=10e-40)
    #pl.ylim(bottom=10e-10)
    pl.xscale('log')
    pl.xlim(left=10e-4)
    #pl.yscale('log')
    pl.legend()
    pl.savefig('tildeepsilon_'+str(a)+'_'+strvec(b_s)+'_n'+strvec(ns)+'.png')
    pl.savefig('tildeepsilon_'+str(a)+'_'+strvec(b_s)+'_n'+strvec(ns)+'.pdf')
    pl.savefig('tildeepsilon_'+str(a)+'_'+strvec(b_s)+'_n'+strvec(ns)+'.ps')

    b=1.
    a_s = [0.0,0.1,0.5,0.8,0.9,0.95]

    pl.clf()
    X = range(0,1000)
    X = [x/1000 for x in X]
    markeverycase  = [(int) (10**(x/3)) for x in range(0,3*3)]

    for n in ns:
        for a in a_s:
            pl.plot(X, [epsilon_right(a, b, x, n, epsilon0=0., epsilon1=1.) for x in X], label='[' + str(a) + ',' + str(b) + '], n=' + str(n),markevery=markeverycase)

    pl.plot(X,[DKW(x,n) for x in X],label='DKW',markevery=markeverycase)


    pl.title('delta -> epsilon_{[a,b]}(n,delta)')
    #pl.ylim(bottom=10e-40)
    #pl.ylim(bottom=10e-10)
    pl.xlim(left=10e-4)
    pl.xscale('log')
    #pl.yscale('log')
    pl.legend()
    pl.savefig('tildeepsilon_'+strvec(a_s)+'_'+str(b)+'_n'+strvec(ns)+'.png')
    pl.savefig('tildeepsilon_'+strvec(a_s)+'_'+str(b)+'_n'+strvec(ns)+'.pdf')
    pl.savefig('tildeepsilon_'+strvec(a_s)+'_'+str(b)+'_n'+strvec(ns)+'.ps')




# # As a function of n. Requires compute_epsilon_left
# def plot_epsilons(delta):
#     N =[2,3,4,5,6,8,10,12,14,16,19,22,25,28,32,36,40,45,50,55,62,69,76,87,96,108,120,132,147,162,180,198,208,230,252,265,288,315,345,380,420,460,500]
#
#     pl.clf()
#     pl.plot(N, [DKW(delta,n) for n in N], label='DKW')
#
#     a=0.
#     b_s = [0.05, 0.1, 0.2, 0.5, 0.9, 1.]
#     for b in b_s:
#         pl.plot(N, load_epsilon_left(a, b, delta), label='[0,' + str(b) + ']')
#
#     pl.ylim(0)
#     pl.xscale('log')
#     #pl.yscale('log')
#     pl.legend()
#     pl.savefig('epsilon_i'+str(a)+'_'+str(b_s)+'_'+str(delta)+'.png')
#     pl.savefig('epsilon_i'+str(a)+'_'+str(b_s)+'_'+str(delta)+'.pdf')


def supx1mx(a,b):
    if (a>0.5):
        return a*(1-a)
    if (b<0.5):
        return b*(1-b)
    return 0.25


def computetimeuniform(a,b,nmax,N,mode='c'):

    C = 2
    q = supx1mx(a, b)

    # Implement the ohoice a) in Cor. 14
    eta = 1 + 1./(np.log(nmax)**0.9)
    delta = 1. / (nmax * np.log(nmax) ** 2)

    if (mode=='b'):
        # Implement the choice b) considered in Cor. 14
        myf = lambda x: np.log(x) + 2.1* np.log(max(np.log(x), 1))
        delta = np.exp(-myf(nmax))
        eta = (myf(nmax) + 1.) / myf(nmax)

    if (mode=='c'):
        # Implement the ohoice c) with eta chosen as in b)
        myf = lambda x: np.log(x) + 2.1* np.log(max(np.log(x), 1))
        eta = (myf(nmax) + 1.) / myf(nmax)
        myg = (nmax+1)*np.log(nmax+1)**2/np.log(2)
        delta = (1./(np.ceil(np.log(nmax)/np.log(eta)))) * (1./(2*myg))


    if (mode=='d'):
        # Implement the ohoice c)
        myf = lambda x: np.log(x) + 2.1* np.log(max(np.log(x), 1))
        eta = (myf(nmax) + 1.) / myf(nmax)
        myg = (nmax+2)*np.log(nmax+2)*(np.log(np.log(nmax+2))**2)/(np.log(np.log(3)))
        delta = (1./(np.ceil(np.log(nmax)/np.log(eta)))) * (1./(2*myg))


    epsilon = np.zeros(len(N))
    tuepsilon = np.zeros(len(N))
    ratios = np.zeros(len(N))
    for i in range(len(N)):
        n = N[i]
        delta_base=1./(nmax*nmax*(nmax+1))
        epsilon[i] = epsilon_right(a, b, delta_base, n, epsilon0=0., epsilon1=1.)
        eps = eta * np.sqrt(n) * epsilon_right(a, b, delta, n, epsilon0=0., epsilon1=1.) + np.sqrt(
            q * eta * (eta - 1) * C / (C - 1))
        tuepsilon[i] = eps / (np.sqrt(n - (eta - 1)))
        ratios[i] = tuepsilon[i] / epsilon[i]
    return ratios

def plottimeuniformepsilon(a,b,mode='a'):
    Ntot = [2,3,4,5,6,8,10,12,14,16,19,22,25,28,32,36,40,45,50,55,62,69,76,87,96,108,120,132,147,162,180,198,208,230,252,265,288,315,345,380,420,460,500,550,600,660,720,780,850,920,1000]
    nmaxs = [5,20,50,100,200,500,1000]


    pl.clf()
    for nmax in nmaxs:
        N = [ n for n in Ntot if n<=nmax]
        ratios=computetimeuniform(a, b, nmax, N,mode)
        pl.plot(N, ratios, label='t='+str(nmax),markevery=markeverycase)

    pl.plot(Ntot,np.ones(len(Ntot)), color='r', linestyle='--', marker='',markevery=markeverycase)
    #pl.plot(N, epsilon, label='single time [' + str(a) + ',' + str(b) + '], d=' + str(delta))

    #pl.plot(N, tuepsilon, label='uniform time [' + str(a) + ',' + str(b) + '], d=' + str(delta))


    pl.title('Peeling vs Union  time uniform bound '+ '[' + str(a) + ',' + str(b) + ']')
    pl.ylim(bottom=10e-10)
    #pl.ylim(bottom=10e-10)
    pl.xscale('log')
    #pl.yscale('log')
    pl.legend()
    s='TimeUniform_'+strvec(nmaxs)+'_'+str(a)+'_'+str(b)+'_mode'+str(mode)
    pl.savefig(s+'.png')
    pl.savefig(s+'.pdf')
    pl.savefig(s+'.ps')

#plottimeuniformepsilon(0,0.7,mode='a')
#plottimeuniformepsilon(0,0.7,mode='b')
#plottimeuniformepsilon(0,0.7,mode='c')
#plottimeuniformepsilon(0,0.7,mode='d')

######################################################################################################
#  Compute Monte Carlo simulations of the probability functions
#
######################################################################################################

def supremum(a, b, f):
    return np.max([f(x / 1000.) for x in range((int) (a * 1000), (int) (b * 1000 + 1))])

def montecarlo_left(epsilons,a, b, n):
    # Nb of MonteCarlo simulations
    M = 10000
    proba = np.zeros(len(epsilons))

    for m in range(M):
        data = np.random.rand(n)
        #print(len(list(filter(lambda x: x<= 0.1, data))))
        f=lambda u: (len(list(filter(lambda x: x<= u, data)))+0.)/(n+0.) -u
        fmax=supremum(a, b, f)
        #print(a,b,fmax)
        for x in range(len(epsilons)):
            if (fmax>epsilons[x]):
                proba[x]+=1./M
    return proba
# TODO: Add dump/load

def montecarlo_right(epsilons,a,b,n):
    # Nb of MonteCarlo simulations
    M = 10000
    proba = np.zeros(len(epsilons))

    for m in range(M):
        data = np.random.rand(n)
        #print(len(list(filter(lambda x: x<= 0.1, data))))
        f=lambda u: u-(len(list(filter(lambda x: x<= u, data)))+0.)/(n+0.)
        fmax=supremum(a, b, f)
        #print(a,b,fmax)
        for x in range(len(epsilons)):
            if (fmax>epsilons[x]):
                proba[x]+=1./M
    return proba


######################################################################################################
#  Plot Monte Carlo simulations of the probability functions
#
######################################################################################################
def plot_proba_MonteCarlo_left(ns):
    a=0.
    bs = [0.05,0.1,0.2,0.5,0.9,1.]
    pl.clf()
    X = range(0,1000)
    X = [x/1000 for x in X]
    markeverycase = (0.0, 0.1)

    for n in ns:
        for b in bs:
            print('MC',n,a,b)
            pl.plot(X, montecarlo_left(X,a, b, n), label='MC [' + str(a) + ',' + str(b) + '], n=' + str(n),markevery=markeverycase)
            #epsilon(a, b, 0.05, n, epsilon0=0., epsilon1=1.)


    pl.title('eps -> Pr( sup U_n-U > eps)')
    pl.ylim(bottom=10e-10)
    #pl.ylim(bottom=10e-10)
    #pl.xscale('log')
    pl.yscale('log')
    pl.legend()
    pl.savefig('MCdelta_'+str(a)+'_'+strvec(bs)+'_n'+strvec(ns)+'.png')
    pl.savefig('MCdelta_'+str(a)+'_'+strvec(bs)+'_n'+strvec(ns)+'.pdf')
    pl.savefig('MCdelta_'+str(a)+'_'+strvec(bs)+'_n'+strvec(ns)+'.ps')

    b=1.
    a_s = [0.0,0.1,0.5,0.8,0.9,0.95]
    pl.clf()
    X = range(0,1000)
    X = [x/1000 for x in X]

    for n in ns:
        for a in a_s:
            print('MC',n,a,b)
            pl.plot(X, montecarlo_left(X,a, b, n), label='MC [' + str(a) + ',' + str(b) + '], n=' + str(n),markevery=markeverycase)
            #epsilon(a, b, 0.05, n, epsilon0=0., epsilon1=1.)


    pl.title('eps -> Pr( sup U_n-U > eps)')
    pl.ylim(bottom=10e-10)
    #pl.ylim(bottom=10e-10)
    #pl.xscale('log')
    pl.yscale('log')
    pl.legend()
    pl.savefig('MCdelta_'+strvec(a_s)+'_'+str(b)+'_n'+strvec(ns)+'.png')
    pl.savefig('MCdelta_'+strvec(a_s)+'_'+str(b)+'_n'+strvec(ns)+'.pdf')
    pl.savefig('MCdelta_'+strvec(a_s)+'_'+str(b)+'_n'+strvec(ns)+'.ps')

def plot_proba_MonteCarlo_right(ns):
    b=1.
    a_s = [0.0,0.1,0.5,0.8,0.9,0.95]
    pl.clf()
    X = range(0,1000)
    X = [x/1000 for x in X]
    markeverycase = (0.0, 0.1)

    for n in ns:
        for a in a_s:
            print('MC',n,a,b)
            pl.plot(X, montecarlo_right(X,a, b, n), label='MC [' + str(a) + ',' + str(b) + '], n=' + str(n),markevery=markeverycase)
            #epsilon(a, b, 0.05, n, epsilon0=0., epsilon1=1.)


    pl.title('eps -> Pr( sup U -U_n > eps)')
    pl.ylim(bottom=10e-10)
    #pl.ylim(bottom=10e-10)
    #pl.xscale('log')
    pl.yscale('log')
    pl.legend()
    pl.savefig('MCtildedelta_'+strvec(a_s)+'_'+str(b)+'_n'+strvec(ns)+'.png')
    pl.savefig('MCtildedelta_'+strvec(a_s)+'_'+str(b)+'_n'+strvec(ns)+'.pdf')
    pl.savefig('MCtildedelta_'+strvec(a_s)+'_'+str(b)+'_n'+strvec(ns)+'.ps')

    a=0.
    bs = [0.05,0.1,0.2,0.5,0.9,1.]
    pl.clf()
    X = range(0,1000)
    X = [x/1000 for x in X]
    markeverycase = (0.0, 0.1)

    for n in ns:
        for b in bs:
            print('MC',n,a,b)
            pl.plot(X, montecarlo_right(X,a, b, n), label='MC [' + str(a) + ',' + str(b) + '], n=' + str(n),markevery=markeverycase)
            #epsilon(a, b, 0.05, n, epsilon0=0., epsilon1=1.)


    pl.title('eps -> Pr( sup U-U_n > eps)')
    pl.ylim(bottom=10e-10)
    #pl.ylim(bottom=10e-10)
    #pl.xscale('log')
    pl.yscale('log')
    pl.legend()
    pl.savefig('MCtildedelta_'+str(a)+'_'+strvec(bs)+'_n'+strvec(ns)+'.png')
    pl.savefig('MCtildedelta_'+str(a)+'_'+strvec(bs)+'_n'+strvec(ns)+'.pdf')
    pl.savefig('MCtildedelta_'+str(a)+'_'+strvec(bs)+'_n'+strvec(ns)+'.ps')



######################################################################################################
#  Demo
#
######################################################################################################
def demo():
    illustrate_left_right()

    # Compute probability values
    for n in [2, 5, 10, 100]:
       plot_proba_left([n])
       plot_proba_right([n])

    # Tabulate boundary values
    print_tabulated_epsilon('epsilon.txt')

    # Print boundary values
    for n in [2, 5, 10, 100]:
        plot_epsilon_left([n])
        plot_epsilon_right([n])

    # Compare probability values to simulated probability values using Monte-Carlo estimates.
    # Computationally intensive due to large  number of samples used in Monte Carlo
    for n in [2,5,10,100]:
        plot_proba_MonteCarlo_left([n])
        plot_proba_MonteCarlo_right([n])


#demo()
#renamefiles()