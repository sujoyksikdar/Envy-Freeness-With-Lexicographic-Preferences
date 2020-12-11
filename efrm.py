import copy
import numpy as np
import pickle
import pulp
import matplotlib
import networkx as nx
from multiprocessing import Pool
from matplotlib import pyplot as plt
from itertools import product

from generate_profiles import gen_mallows


phis = [0.0, 0.25, 0.5, 0.75, 1.0]
ns = [5]
ms = list(range(5, 100 + 5, 5))
Ks = list(range(1, 5+1, 1))
num_instances = 1000
props = ["ef", "ef1", "efx", "mms"]
loc = "MallowsData"
suffix = ""

lns = len(ns)
lms = len(ms)
lps = len(phis)
lks = len(Ks)

smallfont = {"size": 34}
largefont = {"size": 38}
titlefont = {"size": 38}
matplotlib.rc("font", **smallfont)


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
        assert np.min(ranking) == 1
        instances.append((profile, ranking))
        with open(f"{loc}/mallows_n={n},m={m},phi={phi}", "wb") as fo:
            pickle.dump(instances, fo)
    return instances


def load_instances(n, m, phi, num_instances):
    with open(f"{loc}/mallows_n={n},m={m},phi={phi}", "rb") as f:
        instances = pickle.load(f)
    instances = instances[:num_instances]
    return instances


def rmsig(ranking):
    (n, m) = np.shape(ranking)
    sig = dict()
    for j in range(m):
        sig[j] = np.where(ranking[:, j] == np.min(ranking[:, j]))[0].tolist()
    return sig


def isrm(ranking, A):
    (n, m) = np.shape(ranking)
    sig = rmsig(ranking)
    for j in range(m):
        assert np.sum(A[:, j]) == 1
        i = np.where(A[:, j] == 1)[0][0]
        if i not in sig[j]:
            return False
    return True


def isefx(ranking, A):
    (n, m) = np.shape(ranking)
    for i in range(n):
        i_goods = np.where(A[i, :] == 1)
        top_i = min([ranking[i, j] for j in i_goods])[0]
        for k in range(n):
            if i == k:
                continue
            k_goods = np.where(A[k, :] == 1)[0]
            i_prefers_k = list()
            for j in k_goods:
                i_prefers_k.append(int(ranking[i, j] < top_i))
            i_prefers_k = np.sum(i_prefers_k)
            if i_prefers_k > 1:
                return False
    return True


def exists_rm(arg):
    """
    Checks whether the input profile of rankings admits an RM allocation where at most K agents get more than one good
    ------
    Input:
    ------
    ranking: a n x m matrix where the i,j-th entry is i's ranking of good j
    K: an integer 1 <= K <= n
    -------
    Output:
    -------
    exists: True/False
    """
    (ranking, K) = arg
    (n, m) = np.shape(ranking)
    # create the rank maximal allocation signature
    s = rmsig(ranking)

    """
    ILP variables
    """
    # add variables for allocation
    x = pulp.LpVariable.dicts(
        "x",
        [(i, j) for i in range(n) for j in range(m)],
        lowBound=0,
        upBound=1,
        cat="Integer",
    )
    ### x[i,j] = 1, iff good j is assigned to agent i

    e = pulp.LpVariable.dicts(
        "e", [i for i in range(n)], lowBound=0, upBound=1, cat="Integer"
    )
    ### e[i] = 1, iff i is not assigned any good
    z = pulp.LpVariable.dicts(
        "z", [i for i in range(n)], lowBound=0, upBound=1, cat="Integer"
    )
    ### z[i] = 1 if # goods allocated to agent i > 1

    """
    ILP definition and objective
    """
    # define the problem
    model = pulp.LpProblem("rm+efx_existence", pulp.LpMaximize)

    """
    ILP constraints
    """
    # add constraints for rank maximal allocations
    for j in range(m):
        model += pulp.lpSum([x[(i, j)] for i in s[j]]) == 1
        model += pulp.lpSum([x[(i, j)] for i in range(n)]) == 1

    # add constraints to track number of goods allocated to agents
    for i in range(n):
        # set r[i]=m+1 if agent i is not assigned any good
        model += e[i] >= 1 - pulp.lpSum([x[(i, j)] for j in range(m)])
        model += e[i] <= pulp.lpSum([(1 - x[(i, j)]) for j in range(m)]) / m
        model += z[i] >= (pulp.lpSum([x[(i, j)] for j in range(m)]) - 1) / m
        model += z[i] <= pulp.lpSum([x[(i, j)] for j in range(m)]) / 2

    # add constraints to ensure every agent gets at least one good
    """
    for i in range(n):
        model += e[i] <= 0
    """
    # add constraints to ensure at most K agents get more than one goods
    model += pulp.lpSum([z[i] for i in range(n)]) <= K

    """
    Solve
    """
    model.solve(solver=pulp.solvers.GUROBI(msg=False))

    status = model.status
    status = pulp.LpStatusOptimal == model.status
    if status:
        return True
    return False


def exists_efx_rm(ranking):
    """
    Checks whether the input profile of rankings admits an EFX+RM allocation
    ------
    Input:
    ------
    ranking: a n x m matrix where the i,j-th entry is i's ranking of good j
    -------
    Output:
    -------
    exists: True/False
    """
    (n, m) = np.shape(ranking)
    prediag(ranking, "EFX")

    # create the rank maximal allocation signature
    s = rmsig(ranking)

    """
    ILP variables
    """
    # add variables for allocation
    x = pulp.LpVariable.dicts(
        "x",
        [(i, j) for i in range(n) for j in range(m)],
        lowBound=0,
        upBound=1,
        cat="Integer",
    )
    ### x[i,j] = 1, iff good j is assigned to agent i

    # add variables to track agents' top ranked good
    r = pulp.LpVariable.dicts(
        "r", [i for i in range(n)], lowBound=1, upBound=m + 1, cat="Integer"
    )
    ### r[i] = a, if agent i's favorite good among goods allocated to her is ranked a; = m+1 if no good is assigned to i
    # auxiliary variables to track agents' top ranked good
    c = pulp.LpVariable.dicts(
        "c",
        [(i, j) for i in range(n) for j in range(m)],
        lowBound=0,
        upBound=1,
        cat="Integer",
    )
    ### c[i,j] = 0, iff ranking[i,j]<r[i] and x[i,j]=1
    e = pulp.LpVariable.dicts(
        "e", [i for i in range(n)], lowBound=0, upBound=1, cat="Integer"
    )
    ### e[i] = 1, iff i is not assigned any good

    # add variables to track envy relations
    y = pulp.LpVariable.dicts(
        "y",
        [(i, k) for i in range(n) for k in range(n) if not i == k],
        lowBound=0,
        upBound=m,
        cat="Continuous",
    )
    ### y[i,k] = b, if there are b goods allocated to agent k which agent i ranks above the top ranked good allocated to her
    # auxiliary variables to track envy relations
    d = pulp.LpVariable.dicts(
        "d",
        [(i, k, j) for i in range(n) for k in range(n) for j in range(m)],
        lowBound=0,
        upBound=1,
        cat="Integer",
    )
    ### d[i,k,j] = 1, if good j is allocated to agent k and agent i ranks good j above the top ranked good allocated to her
    z = pulp.LpVariable.dicts(
        "z", [i for i in range(n)], lowBound=0, upBound=1, cat="Integer"
    )
    ### z[i] = 1 if # goods allocated to agent i > 1

    """
    ILP definition and objective
    """
    # define the problem
    model = pulp.LpProblem("rm+efx_existence", pulp.LpMaximize)

    # set objective
    model += pulp.lpSum([r[i] for i in range(n)])

    """
    ILP constraints
    """
    # add constraints for rank maximal allocations
    for j in range(m):
        model += pulp.lpSum([x[(i, j)] for i in s[j]]) == 1
        model += pulp.lpSum([x[(i, j)] for i in range(n)]) == 1

    # add constraints to track agents' top ranked goods
    for i in range(n):
        for j in range(m):
            model += r[i] <= ranking[i, j] * x[(i, j)] + (m + 1) * (1 - x[(i, j)])

    for i in range(n):
        for j in range(m):
            model += (
                r[i]
                >= ranking[i, j] * c[(i, j)] + ranking[i, j] * x[(i, j)] - ranking[i, j]
            )
            model += c[(i, j)] >= 1 - x[(i, j)]
            model += c[(i, j)] >= (r[i] - ranking[i, j] * x[(i, j)]) / m
            model += (
                c[(i, j)]
                <= (m + ranking[i, j] * x[(i, j)] + m * (1 - x[(i, j)]) - r[i]) / m
            )
        model += pulp.lpSum([c[(i, j)] for j in range(m)]) == m - (
            pulp.lpSum([x[(i, j)] for j in range(m)]) - 1
        )

        # set r[i]=m+1 if agent i is not assigned any good
        model += e[i] >= 1 - pulp.lpSum([x[(i, j)] for j in range(m)])
        model += e[i] <= pulp.lpSum([(1 - x[(i, j)]) for j in range(m)]) / m
        model += r[i] >= (m + 1) * e[i]

    # add constraints for an EFX allocation
    for i in range(n):
        model += z[i] >= (pulp.lpSum([x[(i, j)] for j in range(m)]) - 1) / m
        model += z[i] <= pulp.lpSum([x[(i, j)] for j in range(m)]) / 2
        for k in range(n):
            if i == k:
                continue
            for j in range(m):
                model += (
                    d[(i, k, j)]
                    >= (r[i] - ranking[i, j] * x[(k, j)] - m * (1 - x[(k, j)])) / m
                )
                model += (
                    d[(i, k, j)]
                    <= (m + r[i] - ranking[i, j] * x[(k, j)] - m * (1 - x[(k, j)])) / m
                )
            model += y[(i, k)] == pulp.lpSum([d[(i, k, j)] for j in range(m)])
            model += y[(i, k)] + z[k] <= 1
            # model += y[(i, k)] <= 1
            # model += pulp.lpSum([x[(k, j)] for j in range(m)]) <= m * (1 - y[(i, k)]) + y[(i, k)]*4

    """
    Solve
    """
    model.solve(solver=pulp.solvers.GUROBI(msg=False))
    # model.solve(solver=pulp.solvers.CPLEX())
    # model.solve()

    status = model.status
    status = pulp.LpStatusOptimal == model.status

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
    ranking: a n x m matrix where the i,j-th entry is i's ranking of good j
    -------
    Output:
    -------
    exists: True/False
    """
    (n, m) = np.shape(ranking)
    prediag(ranking, "EF1")

    # create the rank maximal allocation signature
    s = rmsig(ranking)

    """
    ILP variables
    """
    # add variables for allocation
    x = pulp.LpVariable.dicts(
        "x",
        [(i, j) for i in range(n) for j in range(m)],
        lowBound=0,
        upBound=1,
        cat="Integer",
    )
    ### x[i,j] = 1, iff good j is assigned to agent i

    # add variables to track agents' top ranked good
    r = pulp.LpVariable.dicts(
        "r", [i for i in range(n)], lowBound=1, upBound=m + 1, cat="Integer"
    )
    ### r[i] = a, if agent i's favorite good among goods allocated to her is ranked a; = m+1 if no good is assigned to i
    # auxiliary variables to track agents' top ranked good
    c = pulp.LpVariable.dicts(
        "c",
        [(i, j) for i in range(n) for j in range(m)],
        lowBound=0,
        upBound=1,
        cat="Integer",
    )
    ### c[i,j] = 0, iff ranking[i,j]<r[i] and x[i,j]=1
    e = pulp.LpVariable.dicts(
        "e", [i for i in range(n)], lowBound=0, upBound=1, cat="Integer"
    )
    ### e[i] = 1, iff i is not assigned any good

    # add variables to track envy relations
    y = pulp.LpVariable.dicts(
        "y",
        [(i, k) for i in range(n) for k in range(n) if not i == k],
        lowBound=0,
        upBound=m,
        cat="Continuous",
    )
    ### y[i,k] = b, if there are b goods allocated to agent k which agent i ranks above the top ranked good allocated to her
    # auxiliary variables to track envy relations
    d = pulp.LpVariable.dicts(
        "d",
        [(i, k, j) for i in range(n) for k in range(n) for j in range(m)],
        lowBound=0,
        upBound=1,
        cat="Integer",
    )
    ### d[i,k,j] = 1, if good j is allocated to agent k and agent i ranks good j above the top ranked good allocated to her
    z = pulp.LpVariable.dicts(
        "z", [i for i in range(n)], lowBound=0, upBound=1, cat="Integer"
    )
    ### z[i] = 1 if # goods allocated to agent i > 1

    """
    ILP definition and objective
    """
    # define the problem
    model = pulp.LpProblem("rm+ef1_existence", pulp.LpMaximize)

    # set objective
    model += pulp.lpSum([r[i] for i in range(n)])

    """
    ILP constraints
    """
    # add constraints for rank maximal allocations
    for j in range(m):
        model += pulp.lpSum([x[(i, j)] for i in s[j]]) == 1
        model += pulp.lpSum([x[(i, j)] for i in range(n)]) == 1

    # add constraints to track agents' top ranked goods
    for i in range(n):
        for j in range(m):
            model += r[i] <= ranking[i, j] * x[(i, j)] + (m + 1) * (1 - x[(i, j)])

    for i in range(n):
        for j in range(m):
            model += (
                r[i]
                >= ranking[i, j] * c[(i, j)] + ranking[i, j] * x[(i, j)] - ranking[i, j]
            )
            model += c[(i, j)] >= 1 - x[(i, j)]
            model += c[(i, j)] >= (r[i] - ranking[i, j] * x[(i, j)]) / m
            model += (
                c[(i, j)]
                <= (m + ranking[i, j] * x[(i, j)] + m * (1 - x[(i, j)]) - r[i]) / m
            )
        model += pulp.lpSum([c[(i, j)] for j in range(m)]) == m - (
            pulp.lpSum([x[(i, j)] for j in range(m)]) - 1
        )

        # set r[i]=m+1 if agent i is not assigned any good
        model += e[i] >= 1 - pulp.lpSum([x[(i, j)] for j in range(m)])
        model += e[i] <= pulp.lpSum([(1 - x[(i, j)]) for j in range(m)]) / m
        model += r[i] >= (m + 1) * e[i]

    # add constraints for an EF1 allocation
    for i in range(n):
        model += z[i] >= (pulp.lpSum([x[(i, j)] for j in range(m)]) - 1) / m
        model += z[i] <= pulp.lpSum([x[(i, j)] for j in range(m)]) / 2
    for i in range(n):
        for k in range(n):
            if i == k:
                continue
            for j in range(m):
                model += (
                    d[(i, k, j)]
                    >= (r[i] - ranking[i, j] * x[(k, j)] - m * (1 - x[(k, j)])) / m
                )
                model += (
                    d[(i, k, j)]
                    <= (m + r[i] - ranking[i, j] * x[(k, j)] - m * (1 - x[(k, j)])) / m
                )
            model += y[(i, k)] == pulp.lpSum([d[(i, k, j)] for j in range(m)])
            model += y[(i, k)] <= 1

    """
    Solve
    """
    model.solve(solver=pulp.solvers.GUROBI(msg=False))

    status = model.status
    status = pulp.LpStatusOptimal == model.status

    postdiag(ranking, status, model, x, r, c, e, y, d)

    if status:
        return True
    return False


def exists_ef(ranking):
    """
    Checks whether the input profile of rankings admits an EF1+RM allocation
    ------
    Input:
    ------
    ranking: a n x m matrix where the i,j-th entry is i's ranking of good j
    -------
    Output:
    -------
    exists: True/False
    """
    (n, m) = np.shape(ranking)
    prediag(ranking, "EF")

    """
    ILP variables
    """
    # add variables for allocation
    x = pulp.LpVariable.dicts(
        "x",
        [(i, j) for i in range(n) for j in range(m)],
        lowBound=0,
        upBound=1,
        cat="Integer",
    )
    ### x[i,j] = 1, iff good j is assigned to agent i

    # add variables to track agents' top ranked good
    r = pulp.LpVariable.dicts(
        "r", [i for i in range(n)], lowBound=1, upBound=m + 1, cat="Integer"
    )
    ### r[i] = a, if agent i's favorite good among goods allocated to her is ranked a; = m+1 if no good is assigned to i
    # auxiliary variables to track agents' top ranked good
    c = pulp.LpVariable.dicts(
        "c",
        [(i, j) for i in range(n) for j in range(m)],
        lowBound=0,
        upBound=1,
        cat="Integer",
    )
    ### c[i,j] = 0, iff ranking[i,j]<r[i] and x[i,j]=1
    e = pulp.LpVariable.dicts(
        "e", [i for i in range(n)], lowBound=0, upBound=1, cat="Integer"
    )
    ### e[i] = 1, iff i is not assigned any good

    # add variables to track envy relations
    y = pulp.LpVariable.dicts(
        "y",
        [(i, k) for i in range(n) for k in range(n) if not i == k],
        lowBound=0,
        upBound=m,
        cat="Continuous",
    )
    ### y[i,k] = b, if there are b goods allocated to agent k which agent i ranks above the top ranked good allocated to her
    # auxiliary variables to track envy relations
    d = pulp.LpVariable.dicts(
        "d",
        [(i, k, j) for i in range(n) for k in range(n) for j in range(m)],
        lowBound=0,
        upBound=1,
        cat="Integer",
    )
    ### d[i,k,j] = 1, if good j is allocated to agent k and agent i ranks good j above the top ranked good allocated to her

    """
    ILP definition and objective
    """
    # define the problem
    model = pulp.LpProblem("rm+ef_existence", pulp.LpMaximize)

    # set objective
    model += pulp.lpSum([r[i] for i in range(n)])

    """
    ILP constraints
    """
    # add constraints for rank maximal allocations
    for j in range(m):
        model += pulp.lpSum([x[(i, j)] for i in range(n)]) == 1

    # add constraints to track agents' top ranked goods
    for i in range(n):
        for j in range(m):
            model += r[i] <= ranking[i, j] * x[(i, j)] + (m + 1) * (1 - x[(i, j)])

    for i in range(n):
        for j in range(m):
            model += (
                r[i]
                >= ranking[i, j] * c[(i, j)] + ranking[i, j] * x[(i, j)] - ranking[i, j]
            )
            model += c[(i, j)] >= 1 - x[(i, j)]
            model += c[(i, j)] >= (r[i] - ranking[i, j] * x[(i, j)]) / m
            model += (
                c[(i, j)]
                <= (m + ranking[i, j] * x[(i, j)] + m * (1 - x[(i, j)]) - r[i]) / m
            )
        model += pulp.lpSum([c[(i, j)] for j in range(m)]) == m - (
            pulp.lpSum([x[(i, j)] for j in range(m)]) - 1
        )

        # set r[i]=m+1 if agent i is not assigned any good
        model += e[i] >= 1 - pulp.lpSum([x[(i, j)] for j in range(m)])
        model += e[i] <= pulp.lpSum([(1 - x[(i, j)]) for j in range(m)]) / m
        model += r[i] >= (m + 1) * e[i]

    # add constraints for an EF1 allocation
    for i in range(n):
        for k in range(n):
            if i == k:
                continue
            for j in range(m):
                model += (
                    d[(i, k, j)]
                    >= (r[i] - ranking[i, j] * x[(k, j)] - m * (1 - x[(k, j)])) / m
                )
                model += (
                    d[(i, k, j)]
                    <= (m + r[i] - ranking[i, j] * x[(k, j)] - m * (1 - x[(k, j)])) / m
                )
            model += y[(i, k)] == pulp.lpSum([d[(i, k, j)] for j in range(m)])
            model += y[(i, k)] == 0

    """
    Solve
    """
    model.solve()

    status = model.status
    status = pulp.LpStatusOptimal == model.status

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
    ranking: a n x m matrix where the i,j-th entry is i's ranking of good j
    -------
    Output:
    -------
    exists: True/False
    """
    (n, m) = np.shape(ranking)
    prediag(ranking, "EF")

    # create the rank maximal allocation signature
    s = rmsig(ranking)

    """
    ILP variables
    """
    # add variables for allocation
    x = pulp.LpVariable.dicts(
        "x",
        [(i, j) for i in range(n) for j in range(m)],
        lowBound=0,
        upBound=1,
        cat="Integer",
    )
    ### x[i,j] = 1, iff good j is assigned to agent i

    # add variables to track agents' top ranked good
    r = pulp.LpVariable.dicts(
        "r", [i for i in range(n)], lowBound=1, upBound=m + 1, cat="Integer"
    )
    ### r[i] = a, if agent i's favorite good among goods allocated to her is ranked a; = m+1 if no good is assigned to i
    # auxiliary variables to track agents' top ranked good
    c = pulp.LpVariable.dicts(
        "c",
        [(i, j) for i in range(n) for j in range(m)],
        lowBound=0,
        upBound=1,
        cat="Integer",
    )
    ### c[i,j] = 0, iff ranking[i,j]<r[i] and x[i,j]=1
    e = pulp.LpVariable.dicts(
        "e", [i for i in range(n)], lowBound=0, upBound=1, cat="Integer"
    )
    ### e[i] = 1, iff i is not assigned any good

    # add variables to track envy relations
    y = pulp.LpVariable.dicts(
        "y",
        [(i, k) for i in range(n) for k in range(n) if not i == k],
        lowBound=0,
        upBound=m,
        cat="Continuous",
    )
    ### y[i,k] = b, if there are b goods allocated to agent k which agent i ranks above the top ranked good allocated to her
    # auxiliary variables to track envy relations
    d = pulp.LpVariable.dicts(
        "d",
        [(i, k, j) for i in range(n) for k in range(n) for j in range(m)],
        lowBound=0,
        upBound=1,
        cat="Integer",
    )
    ### d[i,k,j] = 1, if good j is allocated to agent k and agent i ranks good j above the top ranked good allocated to her

    """
    ILP definition and objective
    """
    # define the problem
    model = pulp.LpProblem("rm+ef_existence", pulp.LpMaximize)

    # set objective
    model += pulp.lpSum([r[i] for i in range(n)])

    """
    ILP constraints
    """
    # add constraints for rank maximal allocations
    for j in range(m):
        model += pulp.lpSum([x[(i, j)] for i in s[j]]) == 1
        model += pulp.lpSum([x[(i, j)] for i in range(n)]) == 1

    # add constraints to track agents' top ranked goods
    for i in range(n):
        for j in range(m):
            model += r[i] <= ranking[i, j] * x[(i, j)] + (m + 1) * (1 - x[(i, j)])

    for i in range(n):
        for j in range(m):
            model += (
                r[i]
                >= ranking[i, j] * c[(i, j)] + ranking[i, j] * x[(i, j)] - ranking[i, j]
            )
            model += c[(i, j)] >= 1 - x[(i, j)]
            model += c[(i, j)] >= (r[i] - ranking[i, j] * x[(i, j)]) / m
            model += (
                c[(i, j)]
                <= (m + ranking[i, j] * x[(i, j)] + m * (1 - x[(i, j)]) - r[i]) / m
            )
        model += pulp.lpSum([c[(i, j)] for j in range(m)]) == m - (
            pulp.lpSum([x[(i, j)] for j in range(m)]) - 1
        )

        # set r[i]=m+1 if agent i is not assigned any good
        model += e[i] >= 1 - pulp.lpSum([x[(i, j)] for j in range(m)])
        model += e[i] <= pulp.lpSum([(1 - x[(i, j)]) for j in range(m)]) / m
        model += r[i] >= (m + 1) * e[i]

    # add constraints for an EF1 allocation
    for i in range(n):
        for k in range(n):
            if i == k:
                continue
            for j in range(m):
                model += (
                    d[(i, k, j)]
                    >= (r[i] - ranking[i, j] * x[(k, j)] - m * (1 - x[(k, j)])) / m
                )
                model += (
                    d[(i, k, j)]
                    <= (m + r[i] - ranking[i, j] * x[(k, j)] - m * (1 - x[(k, j)])) / m
                )
            model += y[(i, k)] == pulp.lpSum([d[(i, k, j)] for j in range(m)])
            model += y[(i, k)] == 0

    """
    Solve
    """
    model.solve()

    status = model.status
    status = pulp.LpStatusOptimal == model.status

    postdiag(ranking, status, model, x, r, c, e, y, d)

    if status:
        return True
    return False


def exists_mms_rm(ranking):
    """
    Checks whether the input profile of rankings admits an MMS+RM allocation
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
    agent_nodes = [f'a_{i}' for i in range(n)]
    goods_nodes = [f'g_{j}' for j in range(m)]
    B = nx.Graph()
    B.add_nodes_from(agent_nodes, bipartite=0)
    B.add_nodes_from(goods_nodes, bipartite=1)
    highest_rank = [np.min(ranking[:, j]) for j in range(m)]
    mms_rm_edges = list()
    for i in range(n):
        for j in range(m):
            if ranking[i, j] <= n-1 and ranking[i, j] == highest_rank[j]:
                mms_rm_edges.append((f'a_{i}', f'g_{j}'))
    B.add_edges_from(mms_rm_edges)
    M = nx.bipartite.maximum_matching(B, top_nodes=agent_nodes)
    M = [(f'a_{i}', M[f'a_{i}']) for i in range(n) if f'a_{i}' in M.keys()]
    if len(M) == n:
        return True
    S = list()
    for i in range(n):
        if len([j for j in range(m) if ranking[i, j] >= n and ranking[i, j] == highest_rank[j]]) == m-(n-1):
            S.append(i)
    if len(S) == 0:
        return False
    i_star = S[0]
    bottom_goods = [j for j in range(m) if ranking[i_star, j] >= n]
    top_goods = [j for j in range(m) if j not in bottom_goods]
    ranking_top = ranking[:,[j for j in range(m) if j not in bottom_goods]]
    assert(ranking_top.shape == (n, n-1))
    ranking_star = np.ones((n, n))*n
    ranking_star[:,:-1] = ranking_top
    B = nx.Graph()
    B.add_nodes_from(agent_nodes, bipartite=0)
    goods_nodes = [f'g_{j}' for j in top_goods] + ['g']
    B.add_nodes_from(goods_nodes, bipartite=1)
    mms_rm_edges = list()
    for i in range(n):
        for j in top_goods:
            if ranking[i, j] == highest_rank[j]:
                mms_rm_edges.append((f'a_{i}', f'g_{j}'))
        mms_rm_edges.append((f'a_{i}', 'g'))
    B.add_edges_from(mms_rm_edges)
    M = nx.bipartite.maximum_matching(B, top_nodes=agent_nodes)
    M = [(f'a_{i}', M[f'a_{i}']) for i in range(n) if f'a_{i}' in M.keys()]
    if len(M) == n:
        return True
    return False


def rm_experiments():
    pool = Pool(16)
    for K in Ks:
        for phi in phis:
            for n in ns:
                for m in ms:
                    title = f"rm{K}_mallows_n={n},m={m},phi={phi}{suffix}"
                    print(f"working on {title}")
                    instances = load_instances(n, m, phi, num_instances)
                    instances = [instances[i][1] for i in range(num_instances)]
                    results = pool.map(globals()[f"exists_rm"], product(instances, [K]))
                    instances_exists = np.zeros(num_instances)
                    for i in range(num_instances):
                        instances_exists[i] = int(results[i])
                    with open(f"{loc}/{title}.pickle", "wb") as fo:
                        pickle.dump(instances_exists, fo)
                    print(np.sum(instances_exists))


def ef_experiments():
    pool = Pool(16)
    for phi in phis:
        for n in ns:
            for m in ms:
                title = f"ef_mallows_n={n},m={m},phi={phi}{suffix}"
                print(f"working on {title}")
                instances = load_instances(n, m, phi, num_instances)
                instances = [instances[i][1] for i in range(num_instances)]
                results = pool.map(
                    globals()[f"exists_ef"], instances
                )
                instances_exists = np.zeros(num_instances)
                for i in range(num_instances):
                    instances_exists[i] = int(results[i])
                with open(f"{loc}/{title}.pickle", "wb") as fo:
                    pickle.dump(instances_exists, fo)
                print(np.sum(instances_exists))


def ef_rm_experiments():
    pool = Pool(16)
    for prop in props:
        for phi in phis:
            for n in ns:
                for m in ms:
                    title = f"{prop.lower()}_rm_mallows_n={n},m={m},phi={phi}{suffix}"
                    print(f"working on {title}")
                    instances = load_instances(n, m, phi, num_instances)
                    instances = [instances[i][1] for i in range(num_instances)]
                    results = pool.map(
                        globals()[f"exists_{prop.lower()}_rm"], instances
                    )
                    instances_exists = np.zeros(num_instances)
                    for i in range(num_instances):
                        instances_exists[i] = int(results[i])
                    with open(f"{loc}/{title}.pickle", "wb") as fo:
                        pickle.dump(instances_exists, fo)
                    print(np.sum(instances_exists))


def prediag(ranking, prop):
    if not DEBUG:
        return

    print("-------------------------------------------------------------------")
    print(f"{prop}+RM")
    print("-------------------------------------------------------------------")
    print("ranking\n", ranking)
    (n, m) = np.shape(ranking)
    print(f"top goods {[np.argmin(ranking[i,:]) for i in range(n)]}")


def postdiag(ranking, status, model, x, r, c, e, y, d):
    if not DEBUG:
        return
    # if not status:
        # return

    print(f"status {status}, raw status {model.status}")

    (n, m) = np.shape(ranking)
    s = rmsig(ranking)
    A = np.zeros((n, m))
    # C = np.zeros((n, m))
    # R = np.zeros((n, n))
    Y = np.zeros((n, n))
    for i in range(n):
        for j in range(m):
            # A[i, j] = int(np.round(pulp.value(xvars[(i, j)])))
            A[i, j] = pulp.value(x[(i, j)])
            # C[i, j] = pulp.value(c[(i, j)])
        for k in range(n):
            if i != k:
                Y[i, k] = pulp.value(y[(i, k)])
    """
    for i in range(n):
        for k in range(n):
            R[i, k] = int(pulp.value(r[(i, k)]))
    print(f"R {R}")
    """
    """
    bestranksactual = [
        np.min([ranking[i, j] for j in range(m) if pulp.value(x[(i, j)]) == 1])
        for i in range(n)
    ]
    print(f"best rank actual {bestranksactual}")
    assert bestranks == bestranksactual
    """
    print("allocation\n", A)
    # print("cvars\n", C)
    print("envy\n", Y)
    print("emptiness ", [pulp.value(e[i]) for i in range(n)])
    print("RM signature\n", s)
    print("times maximal ", [np.sum([1 for i in s[j]]) for j in range(m)])
    if status:
        for j in range(m):
            assert np.sum([A[i, j] for i in s[j]]) == 1


def plot_by_ef():
    colors = ["m", "r", "g", "b", "k"]
    markers = ["d", "o", "x", "s", "^"]
    linestyles = [":", "-.", "--", ":", "-"]
    for p in props:
        print(f"{p.upper()}+RM")
        for i in range(lns):
            plt.figure(figsize=(15, 11))
            n = ns[i]
            print(f"n={n}")
            for j in range(lps):
                phi = phis[j]
                y = list()
                for m in ms:
                    with open(
                        f"{loc}/{p}_rm_mallows_n={n},m={m},phi={phi}{suffix}.pickle", "rb"
                    ) as f:
                        r = pickle.load(f)
                    r = np.sum(r) / num_instances
                    y.append(r)
                x = [(m - 5)/5 for m in ms]
                # x = [(m - 5) for m in ms]
                print(p, phi, n, ms)
                print(x)
                print(y)
                if phi in [0.0, 1.0]:
                    phi = int(phi)
                plt.plot(
                    x,
                    y,
                    label=rf"$\phi$={np.round(phi, 2)}",
                    color=colors[j],
                    marker=markers[j],
                    markersize=25,
                    linewidth=3,
                    linestyle=linestyles[j],
                )
            plt.title(f"{p.upper()}+RM", fontdict=titlefont)
            plt.xlim(0, len(ms) - 5)
            plt.xlabel("Number of goods", fontdict=largefont)
            plt.xticks(range(len(ms)), ms)
            plt.ylim(0.0, 1.02)
            plt.ylabel(f"Fraction of instances", fontdict=largefont)
            plt.grid()
            if p in ["ef"]:
                plt.legend(loc="upper left")
            else:
                plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(f"{loc}/mallows_{p.upper()}+RM_n={n}_existence{suffix}.png")
            plt.show()


def plot_ef_rm():
    colors = ["m", "r", "g", "b", "k"]
    markers = ["d", "o", "x", "s", "^"]
    linestyles = [":", "-.", "--", ":", "-"]
    for j in range(lps):
        plt.figure(figsize=(20, 11))
        n = 5
        yrm = list()
        yefrm = list()
        yefxrm = list()
        ynefrm = list()
        for m in ms:
            phi = phis[j]
            with open(
                f"{loc}/efx_rm_mallows_n={n},m={m},phi={phi}{suffix}.pickle", "rb"
            ) as f:
                r_efxrm = pickle.load(f)
            with open(
                f"{loc}/ef_rm_mallows_n={n},m={m},phi={phi}{suffix}.pickle", "rb"
            ) as f:
                r_efrm = pickle.load(f)
            with open(
                f"{loc}/ef_mallows_n={n},m={m},phi={phi}{suffix}.pickle", "rb"
            ) as f:
                r_ef = pickle.load(f)
            with open(
                f"{loc}/rm4_mallows_n={n},m={m},phi={phi}{suffix}.pickle", "rb"
            ) as f:
                r_rm = pickle.load(f)
            r_efxrm = np.sum(r_efxrm) / num_instances
            r_efrm = np.sum(r_efrm) / num_instances
            r_nef = [1 - r_ef[i] for i in range(len(r_ef))]
            r_nefrm = list()
            for l in range(len(r_ef)):
                if r_nef[l] == 1 and r_rm[l] == 1:
                    r_nefrm.append(1)
                else:
                    r_nefrm.append(0)
            r_nefrm = np.sum(r_nefrm) / num_instances
            yefxrm.append(r_efxrm)
            yefrm.append(r_efrm)
            ynefrm.append(r_nefrm + r_efrm)
        x = [(m - 5)/5 for m in ms]
        if phi in [0.0, 1.0]:
            phi = int(phi)
        print(phi)
        print(x)
        print(yefxrm)
        print(yefrm)
        print(ynefrm)
        plt.plot(
            x,
            yefxrm,
            # label=rf"EF+RM,$\phi$={phi}",
            label=rf"EFX+RM",
            color='k',
            marker='x',
            markersize=25,
            linewidth=3,
            linestyle=":",
        )
        plt.plot(
            x,
            yefrm,
            # label=rf"EF+RM,$\phi$={phi}",
            label=rf"EF+RM",
            color='k',
            marker='o',
            markersize=25,
            linewidth=3,
            linestyle="--",
        )
        plt.plot(
            x,
            ynefrm,
            # label=rf"NoEF,$\phi$={phi}",
            label=rf"NonEF+GoodRM",
            color='k',
            marker='o',
            markersize=25,
            linewidth=3,
            linestyle="-",
        )
        plt.fill_between(x, ynefrm, yefrm, interpolate=True, facecolor='k', alpha=0.2)
        # plt.title(f"RM vs EFX+RM", fontdict=titlefont)
        plt.title(rf'$\phi$={phi}')
        plt.xlim(0, len(ms) - 5)
        plt.xlabel("Number of goods", fontdict=largefont)
        plt.xticks(range(len(ms)), ms)
        plt.ylim(0.0, 1.02)
        plt.ylabel(f"Fraction of instances", fontdict=largefont)
        plt.grid()
        # plt.legend(loc="lower left")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(f"{loc}/mallows_EF_RM_EFRM_n={n}_phi={phi}{suffix}.png")
        plt.show()


def main():
    # """
    # generate instances
    for n in ns:
        for m in ms:
            for phi in phis:
                instances = generate_instances(n, m, phi, num_instances)
    # """

    # """
    # run experiments
    ef_rm_experiments()
    rm_experiments()
    ef_experiments()
    # """

    # """
    plot_by_ef()
    plot_ef_rm()
    # """


if __name__ == "__main__":
    main()
    print("done")
