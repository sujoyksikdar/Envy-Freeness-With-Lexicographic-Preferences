import copy
import numpy as np
import pickle
import pulp
import sys
import matplotlib
from multiprocessing import Pool
from matplotlib import pyplot as plt
import seaborn as sns

from generate_profiles import *


phis = [0.0, 0.25, 0.5, 0.75, 1.0]
ns = [5]
ms = list(range(5, 5*3 + 1, 1))
num_instances = 100
props = ['ef', 'ef1', 'efx']
loc = 'ef_rm_experiments'

lns = len(ns)
lms = len(ms)
lps = len(phis)

smallfont = {
'size': 34
}
largefont = {
'size': 38
}
titlefont = {
'size': 38
}
matplotlib.rc('font', **smallfont)


DEBUG = False
def debug(*args):
    if DEBUG:
        print(*args)


def generate_instances(n, m, phi, num_instances):
    instances = list()

    candmap = dict((j, j) for j in range(m))
    ref = list(range(m))
    for i in range(num_instances):
        rankmap, rankmapcounts = gen_mallows(n, candmap, [1], [phi], [ref])
        # print(rankmap)
        # print(rankmapcounts)
        profile = list()
        ranking = list()
        for r in range(len(rankmap)):
            count = rankmapcounts[r]
            for c in range(count):
                pref = np.zeros(m, dtype=int)
                ranks = np.zeros(m, dtype=int)
                for j in range(m):
                    rank = rankmap[r][j] - 1
                    pref[rank] = j
                    ranks[j] = rank
                profile.append(pref)
                ranking.append(ranks)
        profile = np.array(profile)
        ranking = np.array(ranking)
        if np.min(ranking) <= 0:
            minval = np.min(ranking)
            ranking += -minval + 1
        elif np.min(ranking) > 1:
            minval = np.min(ranking)
            ranking -= minval + 1
        assert(np.min(ranking) == 1)
        instances.append((profile, ranking))
        with open(f'{loc}/mallows_n={n},m={m},phi={phi}', 'wb') as fo:
            pickle.dump(instances, fo)
    return instances


def load_instances(n, m, phi, num_instances):
    with open(f'{loc}/mallows_n={n},m={m},phi={phi}', 'rb') as f:
        instances = pickle.load(f)
    return instances


def rmsig(ranking):
    (n, m) = np.shape(ranking)
    sig = dict()
    for j in range(m):
        sig[j] = np.where(ranking[:, j] == np.min(ranking[:, j]))[0].tolist()
    return sig


def isrm(ranking, A):
    sig = rmsig(ranking)
    for j in range(m):
        assert(np.sum(A[:, j]) == 1)
        i = np.where(A[:, j] == 1)[0][0]
        if not i in sig[j]:
            return False
    return True


def isefx(ranking, A):
    for i in range(n):
        i_items = np.where(A[i, :] == 1)
        top_i = min([ranking[i,j] for j in i_items])[0]
        for k in range(n):
            if i == k:
                continue
            k_items = np.where(A[k, :] == 1)[0]
            i_prefers_k = list()
            for j in k_items:
                i_prefers_k.append(int(ranking[i, j] < top_i))
            i_prefers_k = np.sum(i_prefers_k)
            if i_prefers_k > 1:
                return False
    return True


def rqsd(in_ranking):
    ranking = copy.deepcopy(in_ranking)
    (n, m) = np.shape(ranking)

    A = np.zeros((n,m))

    remaining = list(range(m))
    for i in range(n):
        top = np.argmin(ranking[i, :])
        assert(top in remaining)
        A[i, top] = 1
        remaining.remove(top)
        ranking[:, top] = 2*m
    for j in remaining:
        A[n-1, j] = 1
    for j in range(m):
        assert(np.sum(A[:, j]) == 1)
    return A


def exists_rm_forall(ranking):
    (n, m) = np.shape(ranking)

    rmsig = dict()
    for j in range(m):
        rmsig[j] = np.where(ranking[:, j] == np.min(ranking[:, j]))[0].tolist()
    debug('rmsig\n', rmsig)

    # add variables for allocation
    xvars = pulp.LpVariable.dicts('x', [(i, j) for i in range(n) for j in range(m)], lowBound=0, upBound=1, cat='Integer')

    # add variables for the rank of the top item assigned to each agent
    rvars = pulp.LpVariable.dicts('r', [i for i in range(n)], lowBound=1, upBound=m, cat='Integer')
    cvars = pulp.LpVariable.dicts('c', [(i, j) for i in range(n) for j in range(m)], lowBound=0, upBound=1, cat='Integer')

    # define the problem
    model = pulp.LpProblem('rm+efx_existence', pulp.LpMaximize)

    # add allocation constraints
    for j in range(m):
        model += pulp.lpSum([xvars[(i, j)] for i in range(n)]) == 1
    for i in range(n):
        model += pulp.lpSum([xvars[(i, j)] for j in range(m)]) >= 1
    # add rank maximality constraints
    for j in range(m):
        model += pulp.lpSum([xvars[(i, j)] for i in rmsig[j]]) == 1

    # add constraints to track each agents' top ranked item
    for i in range(n):
        for j in range(m):
            model += rvars[i] <= ranking[i, j]*xvars[(i, j)] + m*(1-xvars[(i,j)])
            model += cvars[(i, j)] >= (ranking[i, j]*xvars[(i,j)] + m*(1 - xvars[(i, j)]) - rvars[i])/m
            model += cvars[(i, j)] <= (m + ranking[i, j]*xvars[(i, j)] + m*(1 - xvars[(i, j)]) - rvars[i])/m
        model += pulp.lpSum([cvars[(i, j)] for j in range(m)]) == m
        model += rvars[i] # adding this to the objective. forces rvars to be set to highest rank

    model.solve()

    status = model.status
    status = (pulp.LpStatusOptimal == model.status)

    debug(f'status {status}, raw status {model.status}')

    if status or DEBUG:
        A = np.zeros((n, m))
        C = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                # A[i, j] = int(np.round(pulp.value(xvars[(i, j)])))
                A[i, j] = pulp.value(xvars[(i, j)])
                C[i, j] = pulp.value(cvars[(i, j)])
        debug(f'best ranked items {[pulp.value(rvars[i]) for i in range(n)]}')
        debug('allocation\n' , A)
        debug('cvars\n' , C)
        debug('ranking\n', ranking)
        # sanity checks
        for j in range(m):
            assert(np.sum([A[i, j] for i in rmsig[j]]) == 1)
        if status:
            return True
    return False


def exists_efx_rm(ranking):
    """
    Checks whether the input profile of rankings admits an EFX+RM allocation
    ------
    Input:
    ------
    ranking: a n x m matrix where the i,j-th entry is i's ranking of item j
    -------
    Output:
    -------
    exists: True/False
    """
    (n, m) = np.shape(ranking)
    prediag(ranking, 'EFX')

    # create the rank maximal allocation signature
    s = rmsig(ranking)

    """
    ILP variables
    """
    # add variables for allocation
    x = pulp.LpVariable.dicts('x', [(i, j) for i in range(n) for j in range(m)], lowBound=0, upBound=1, cat='Integer')
    ### x[i,j] = 1, iff item j is assigned to agent i

    # add variables to track agents' top ranked item
    r = pulp.LpVariable.dicts('r', [i for i in range(n)], lowBound=1, upBound=m+1, cat='Integer')
    ### r[i] = a, if agent i's favorite item among items allocated to her is ranked a; = m+1 if no item is assigned to i
    # auxiliary variables to track agents' top ranked item
    c = pulp.LpVariable.dicts('c', [(i,j) for i in range(n) for j in range(m)], lowBound=0, upBound=1, cat='Integer')
    ### c[i,j] = 0, iff ranking[i,j]<r[i] and x[i,j]=1
    e = pulp.LpVariable.dicts('e', [i for i in range(n)], lowBound=0, upBound=1, cat='Integer')
    ### e[i] = 1, iff i is not assigned any item

    # add variables to track envy relations
    y = pulp.LpVariable.dicts('y', [(i,k) for i in range(n) for k in range(n) if not i == k],
                              lowBound=0, upBound=m, cat='Continuous')
    ### y[i,k] = b, if there are b items allocated to agent k which agent i ranks above the top ranked item allocated to her
    # auxiliary variables to track envy relations
    d = pulp.LpVariable.dicts('d', [(i,k,j) for i in range(n) for k in range(n) for j in range(m)],
                              lowBound=0, upBound=1, cat='Integer')
    ### d[i,k,j] = 1, if item j is allocated to agent k and agent i ranks item j above the top ranked item allocated to her

    """
    ILP definition and objective
    """
    # define the problem
    model = pulp.LpProblem('rm+efx_existence', pulp.LpMaximize)

    # set objective
    model += pulp.lpSum([r[i] for i in range(n)])

    """
    ILP constraints
    """
    # add constraints for rank maximal allocations
    for j in range(m):
        model += pulp.lpSum([x[(i,j)] for i in s[j]]) == 1
        model += pulp.lpSum([x[(i,j)] for i in range(n)]) == 1

    # add constraints to track agents' top ranked items
    for i in range(n):
        for j in range(m):
            model += r[i] <= ranking[i,j]*x[(i,j)] + (m+1)*(1-x[(i,j)])

    for i in range(n):
        for j in range(m):
            model += r[i] >= ranking[i,j]*c[(i,j)] + ranking[i,j]*x[(i,j)] - ranking[i,j]
            model += c[(i,j)] >= 1-x[(i,j)]
            model += c[(i,j)] >= (r[i] - ranking[i,j]*x[(i,j)])/m
            model += c[(i,j)] <= (m + ranking[i,j]*x[(i,j)] + m*(1-x[(i,j)]) - r[i])/m
        model += pulp.lpSum([c[(i,j)] for j in range(m)]) == m - (pulp.lpSum([x[(i,j)] for j in range(m)]) - 1)

        # set r[i]=m+1 if agent i is not assigned any item
        model += e[i] >= 1 - pulp.lpSum([x[(i,j)] for j in range(m)])
        model += e[i] <= pulp.lpSum([(1-x[(i,j)]) for j in range(m)])/m
        model += r[i] >= (m+1)*e[i]

    # add constraints for an EFX allocation
    for i in range(n):
        for k in range(n):
            if i == k:
                continue
            for j in range(m):
                model += d[(i,k,j)] >= (r[i] - ranking[i,j]*x[(k,j)] - m*(1-x[(k,j)]))/m
                model += d[(i,k,j)] <= (m + r[i] - ranking[i,j]*x[(k,j)] - m*(1-x[(k,j)]))/m
            model += y[(i,k)] == pulp.lpSum([d[(i,k,j)] for j in range(m)])
            model += y[(i,k)] <= 1

    """
    Solve
    """
    model.solve()

    status = model.status
    status = (pulp.LpStatusOptimal == model.status)

    postdiag(ranking, status, model, x, r, c, e, y, d)
    if status:
        return True
    return False


def exists_ef1_rm(ranking):
    """
    Checks whether the input profile of rankings admits an EF1+RM allocation
    ------
    Input:
    ------
    ranking: a n x m matrix where the i,j-th entry is i's ranking of item j
    -------
    Output:
    -------
    exists: True/False
    """
    (n, m) = np.shape(ranking)
    prediag(ranking, 'EF1')

    # create the rank maximal allocation signature
    s = rmsig(ranking)

    """
    ILP variables
    """
    # add variables for allocation
    x = pulp.LpVariable.dicts('x', [(i, j) for i in range(n) for j in range(m)], lowBound=0, upBound=1, cat='Integer')
    ### x[i,j] = 1, iff item j is assigned to agent i

    # add variables to track agents' top ranked item
    r = pulp.LpVariable.dicts('r', [i for i in range(n)], lowBound=1, upBound=m+1, cat='Integer')
    ### r[i] = a, if agent i's favorite item among items allocated to her is ranked a; = m+1 if no item is assigned to i
    # auxiliary variables to track agents' top ranked item
    c = pulp.LpVariable.dicts('c', [(i,j) for i in range(n) for j in range(m)], lowBound=0, upBound=1, cat='Integer')
    ### c[i,j] = 0, iff ranking[i,j]<r[i] and x[i,j]=1
    e = pulp.LpVariable.dicts('e', [i for i in range(n)], lowBound=0, upBound=1, cat='Integer')
    ### e[i] = 1, iff i is not assigned any item

    # add variables to track envy relations
    y = pulp.LpVariable.dicts('y', [(i,k) for i in range(n) for k in range(n) if not i == k],
                              lowBound=0, upBound=m, cat='Continuous')
    ### y[i,k] = b, if there are b items allocated to agent k which agent i ranks above the top ranked item allocated to her
    # auxiliary variables to track envy relations
    d = pulp.LpVariable.dicts('d', [(i,k,j) for i in range(n) for k in range(n) for j in range(m)],
                              lowBound=0, upBound=1, cat='Integer')
    ### d[i,k,j] = 1, if item j is allocated to agent k and agent i ranks item j above the top ranked item allocated to her
    z = pulp.LpVariable.dicts('z', [i for i in range(n)], lowBound=0, upBound=1, cat='Integer')

    """
    ILP definition and objective
    """
    # define the problem
    model = pulp.LpProblem('rm+ef1_existence', pulp.LpMaximize)

    # set objective
    model += pulp.lpSum([r[i] for i in range(n)])

    """
    ILP constraints
    """
    # add constraints for rank maximal allocations
    for j in range(m):
        model += pulp.lpSum([x[(i,j)] for i in s[j]]) == 1
        model += pulp.lpSum([x[(i,j)] for i in range(n)]) == 1

    # add constraints to track agents' top ranked items
    for i in range(n):
        for j in range(m):
            model += r[i] <= ranking[i,j]*x[(i,j)] + (m+1)*(1-x[(i,j)])

    for i in range(n):
        for j in range(m):
            model += r[i] >= ranking[i,j]*c[(i,j)] + ranking[i,j]*x[(i,j)] - ranking[i,j]
            model += c[(i,j)] >= 1-x[(i,j)]
            model += c[(i,j)] >= (r[i] - ranking[i,j]*x[(i,j)])/m
            model += c[(i,j)] <= (m + ranking[i,j]*x[(i,j)] + m*(1-x[(i,j)]) - r[i])/m
        model += pulp.lpSum([c[(i,j)] for j in range(m)]) == m - (pulp.lpSum([x[(i,j)] for j in range(m)]) - 1)

        # set r[i]=m+1 if agent i is not assigned any item
        model += e[i] >= 1 - pulp.lpSum([x[(i,j)] for j in range(m)])
        model += e[i] <= pulp.lpSum([(1-x[(i,j)]) for j in range(m)])/m
        model += r[i] >= (m+1)*e[i]

    # add constraints for an EF1 allocation
    for i in range(n):
        model += z[i] >= (pulp.lpSum([x[(i,j)] for j in range(m)]) - 1)/m
        model += z[i] <= pulp.lpSum([x[(i,j)] for j in range(m)])/2
    for i in range(n):
        for k in range(n):
            if i == k:
                continue
            for j in range(m):
                model += d[(i,k,j)] >= (r[i] - ranking[i,j]*x[(k,j)] - m*(1-x[(k,j)]))/m
                model += d[(i,k,j)] <= (m + r[i] - ranking[i,j]*x[(k,j)] - m*(1-x[(k,j)]))/m
            model += y[(i,k)] == pulp.lpSum([d[(i,k,j)] for j in range(m)])
            model += y[(i,k)] <= pulp.lpSum([x[(k,j)] for j in range(m)]) - z[k]

    """
    Solve
    """
    model.solve()

    status = model.status
    status = (pulp.LpStatusOptimal == model.status)

    postdiag(ranking, status, model, x, r, c, e, y, d)

    if status:
        return True
    return False


def exists_ef_rm(ranking):
    """
    Checks whether the input profile of rankings admits an EF1+RM allocation
    ------
    Input:
    ------
    ranking: a n x m matrix where the i,j-th entry is i's ranking of item j
    -------
    Output:
    -------
    exists: True/False
    """
    (n, m) = np.shape(ranking)
    prediag(ranking, 'EF')

    # create the rank maximal allocation signature
    s = rmsig(ranking)

    """
    ILP variables
    """
    # add variables for allocation
    x = pulp.LpVariable.dicts('x', [(i, j) for i in range(n) for j in range(m)], lowBound=0, upBound=1, cat='Integer')
    ### x[i,j] = 1, iff item j is assigned to agent i

    # add variables to track agents' top ranked item
    r = pulp.LpVariable.dicts('r', [i for i in range(n)], lowBound=1, upBound=m+1, cat='Integer')
    ### r[i] = a, if agent i's favorite item among items allocated to her is ranked a; = m+1 if no item is assigned to i
    # auxiliary variables to track agents' top ranked item
    c = pulp.LpVariable.dicts('c', [(i,j) for i in range(n) for j in range(m)], lowBound=0, upBound=1, cat='Integer')
    ### c[i,j] = 0, iff ranking[i,j]<r[i] and x[i,j]=1
    e = pulp.LpVariable.dicts('e', [i for i in range(n)], lowBound=0, upBound=1, cat='Integer')
    ### e[i] = 1, iff i is not assigned any item

    # add variables to track envy relations
    y = pulp.LpVariable.dicts('y', [(i,k) for i in range(n) for k in range(n) if not i == k],
                              lowBound=0, upBound=m, cat='Continuous')
    ### y[i,k] = b, if there are b items allocated to agent k which agent i ranks above the top ranked item allocated to her
    # auxiliary variables to track envy relations
    d = pulp.LpVariable.dicts('d', [(i,k,j) for i in range(n) for k in range(n) for j in range(m)],
                              lowBound=0, upBound=1, cat='Integer')
    ### d[i,k,j] = 1, if item j is allocated to agent k and agent i ranks item j above the top ranked item allocated to her

    """
    ILP definition and objective
    """
    # define the problem
    model = pulp.LpProblem('rm+ef_existence', pulp.LpMaximize)

    # set objective
    model += pulp.lpSum([r[i] for i in range(n)])

    """
    ILP constraints
    """
    # add constraints for rank maximal allocations
    for j in range(m):
        model += pulp.lpSum([x[(i,j)] for i in s[j]]) == 1
        model += pulp.lpSum([x[(i,j)] for i in range(n)]) == 1

    # add constraints to track agents' top ranked items
    for i in range(n):
        for j in range(m):
            model += r[i] <= ranking[i,j]*x[(i,j)] + (m+1)*(1-x[(i,j)])

    for i in range(n):
        for j in range(m):
            model += r[i] >= ranking[i,j]*c[(i,j)] + ranking[i,j]*x[(i,j)] - ranking[i,j]
            model += c[(i,j)] >= 1-x[(i,j)]
            model += c[(i,j)] >= (r[i] - ranking[i,j]*x[(i,j)])/m
            model += c[(i,j)] <= (m + ranking[i,j]*x[(i,j)] + m*(1-x[(i,j)]) - r[i])/m
        model += pulp.lpSum([c[(i,j)] for j in range(m)]) == m - (pulp.lpSum([x[(i,j)] for j in range(m)]) - 1)

        # set r[i]=m+1 if agent i is not assigned any item
        model += e[i] >= 1 - pulp.lpSum([x[(i,j)] for j in range(m)])
        model += e[i] <= pulp.lpSum([(1-x[(i,j)]) for j in range(m)])/m
        model += r[i] >= (m+1)*e[i]

    # add constraints for an EF1 allocation
    for i in range(n):
        for k in range(n):
            if i == k:
                continue
            for j in range(m):
                model += d[(i,k,j)] >= (r[i] - ranking[i,j]*x[(k,j)] - m*(1-x[(k,j)]))/m
                model += d[(i,k,j)] <= (m + r[i] - ranking[i,j]*x[(k,j)] - m*(1-x[(k,j)]))/m
            model += y[(i,k)] == pulp.lpSum([d[(i,k,j)] for j in range(m)])
            model += y[(i,k)] == 0

    """
    Solve
    """
    model.solve()

    status = model.status
    status = (pulp.LpStatusOptimal == model.status)

    postdiag(ranking, status, model, x, r, c, e, y, d)

    if status:
        return True
    return False


def ef_rm_experiments():
    pool = Pool(6)
    for prop in props:
        for phi in phis:
            for n in ns:
                for m in ms:
                    title = f'{prop.lower()}_rm_mallows_n={n},m={m},phi={phi}'
                    print(f'working on {title}')
                    instances = load_instances(n, m, phi, num_instances)
                    instances = [instances[i][1] for i in range(num_instances)]
                    results = pool.map(globals()[f'exists_{prop.lower()}_rm'], instances)
                    instances_exists = np.zeros(num_instances)
                    for i in range(num_instances):
                        instances_exists[i] = int(results[i])
                    with open(f'{loc}/{title}.pickle', 'wb') as fo:
                        pickle.dump(instances_exists, fo)
                    print(np.sum(instances_exists))


def prediag(ranking, prop):
    if not DEBUG:
        return

    print('-------------------------------------------------------------------')
    print(f'{prop}+RM')
    print('-------------------------------------------------------------------')
    print('ranking\n', ranking)
    (n, m) = np.shape(ranking)
    print(f'top items {[np.argmin(ranking[i,:]) for i in range(n)]}')


def postdiag(ranking, status, model, x, r, c, e, y, d):
    if not DEBUG:
        return
    if not status:
        return

    print(f'status {status}, raw status {model.status}')

    (n, m) = np.shape(ranking)
    s = rmsig(ranking)
    A = np.zeros((n, m))
    C = np.zeros((n, m))
    Y = np.zeros((n, n))
    for i in range(n):
        for j in range(m):
            # A[i, j] = int(np.round(pulp.value(xvars[(i, j)])))
            A[i, j] = pulp.value(x[(i, j)])
            C[i, j] = pulp.value(c[(i, j)])
        for k in range(n):
            if i != k:
                Y[i, k] = pulp.value(y[(i, k)])
    bestranks = [int(pulp.value(r[i])) for i in range(n)]
    print(f'best ranks {bestranks}')
    bestranksactual = [np.min([ranking[i, j] for j in range(m) if pulp.value(x[(i,j)]) == 1]) for i in range(n)]
    print(f'best rank actual {bestranksactual}')
    assert(bestranks == bestranksactual)
    print('allocation\n' , A)
    print('cvars\n' , C)
    print('envy\n', Y)
    print('emptiness ', [pulp.value(e[i]) for i in range(n)])
    print('s: ', s)
    print('times allocated ', [np.sum([pulp.value(x[(i,j)]) for i in s[j]]) for j in range(m)])
    print('times maximal ', [np.sum([1 for i in s[j]]) for j in range(m)])
    for j in range(m):
        assert(np.sum([A[i, j] for i in s[j]]) == 1)


def rqsd_experiments():
    for phi in phis:
        for n in ns:
            for m in ms:
                title = f'mallows_rm_rqsd_n={n},m={m},phi={phi}'
                instances = load_instances(n, m, phi, num_instances)
                instances_rm = np.zeros(num_instances)
                for i in range(num_instances):
                    (profile, ranking) = instances[i]
                    A = rqsd(ranking)
                    assert(isefx(ranking, A))
                    rm = isrm(ranking, A)
                    instances_rm[i] = rm
                print(f'{title}, rm={np.sum(instances_rm)}')
                with open(f'{loc}/{title}.pickle', 'wb') as fo:
                    pickle.dump(instances_rm, fo)


def plot_by_phi():
    for n in ns:
        title = f'mallows_n={n}'
        lms = len(ms)
        lps = len(phis)
        H = np.zeros((lps, lms))
        for i in range(lps):
            phi = phis[i]
            for j in range(lms):
                m = ms[j]
                with open(f'mallows_n={n},m={m},phi={phi}.pickle', 'rb') as f:
                    r = pickle.load(f)
                r = np.sum(r)
                H[i, j] = r
        print('n =', n, '\n', H)

        fig = plt.figure()
        sns.heatmap(H, cmap='coolwarm')
        plt.title(f'# agents = {n}')
        plt.xlabel('# items')
        plt.ylabel(r'$\phi$ Mallows model')
        plt.xticks([j+0.5 for j in range(lms)], ms)
        plt.yticks([i+0.5 for i in range(lps)], np.round(phis,1), rotation=0)
        for im in fig.gca().get_images():
            im.clim(vmin=np.min(H), vmax=100)
        plt.savefig(title+'.png')


def plot_by_ef():
    colors = ['r', 'g', 'b', 'k']
    markers = ['o','x','s','^']
    linestyles = ['-', '--', '-.', ':']
    for p in props:
        print(f'{p.upper()}+RM')
        for i in range(lns):
            plt.figure(figsize=(15, 9))
            n = ns[i]
            print(f'n={n}')
            for j in range(lps):
                phi = phis[j]
                y = list()
                for m in ms:
                    with open(f'{loc}/{p}_rm_mallows_n={n},m={m},phi={phi}.pickle', 'rb') as f:
                        r = pickle.load(f)
                    r = np.sum(r)/100.
                    y.append(r)
                x = [m-5 for m in ms]
                plt.plot(x, y, label=rf'$\phi$={phi}', color=colors[j], marker=markers[j], markersize=25, linewidth=3, linestyle=linestyles[j])
            plt.title(f'Existence of {p.upper()}+RM allocations \n (n={n}, Mallows model)', fontdict=titlefont)
            plt.xlim(0, len(ms)-5)
            plt.xlabel('Number of goods', fontdict=largefont)
            plt.xticks(range(len(ms)), ms)
            plt.ylim(.50, 1.02)
            plt.ylabel(f'Fraction of instances', fontdict=largefont)
            plt.grid()
            plt.legend(loc='lower right')
            plt.tight_layout()
            plt.savefig(f'{loc}/mallows_{p.upper()}+RM_n={n}_existence.png')
            plt.show()


def is_ef_rm():
    for phi in phis:
        for n in ns:
            for m in ms:
                with open(f'{loc}/efx_rm_mallows_n={n},m={m},phi={phi}.pickle', 'rb') as f:
                    efx_rm = np.array(pickle.load(f))
                with open(f'{loc}/ef1_rm_mallows_n={n},m={m},phi={phi}.pickle', 'rb') as f:
                    ef1_rm = np.array(pickle.load(f))
                with open(f'{loc}/ef_rm_mallows_n={n},m={m},phi={phi}.pickle', 'rb') as f:
                    ef_rm = np.array(pickle.load(f))
                print(phi, n, m)
                efx_rm_is_ef_rm = np.all((efx_rm == ef_rm) == True)
                ef1_rm_is_ef_rm = np.all((ef1_rm == ef_rm) == True)
                print(f'efx {efx_rm_is_ef_rm}')
                print(f'ef1 {ef1_rm_is_ef_rm}')
                assert(efx_rm_is_ef_rm)
                assert(ef1_rm_is_ef_rm)


def main():
    """
    # generate instances
    for n in ns:
        for m in ms:
            for phi in phis:
                instances = generate_instances(n, m, phi, num_instances)
    """
    # """
    # run experiments
    ef_rm_experiments()
    # """
    """
    plot_by_ef()
    """
    # is_ef_rm()


if __name__ == '__main__':
    main()
    print('done')
