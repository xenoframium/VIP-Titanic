#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 08:02:32 2020

@author: tsandhu
"""

import random
import operator
import itertools

import numpy as np
import matplotlib.pyplot as plt

from deap import base
from deap import creator
from deap import tools
from deap import gp

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import networkx as nx
import pygraphviz as pgv

"""----Data Processing----"""


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data.columns[train_data.isna().any()].tolist()

train_data.set_index(keys=['PassengerId'], drop=True, inplace=True)

test_data.set_index(keys=['PassengerId'], drop=True, inplace=True)

train_nan_map = {'Fare': train_data['Fare'].mean(), 'Embarked': train_data['Embarked'].mode()[0]}
test_nan_map = {'Fare': test_data['Fare'].mean(), 'Embarked': test_data['Embarked'].mode()[0]}

train_data.fillna(value=train_nan_map, inplace=True)
test_data.fillna(value=test_nan_map, inplace=True)

#Southampton Cherbourg Queenstown

columns_map = {'Sex': {'male': 0, 'female': 1}}

train_data.replace(columns_map, inplace=True)
test_data.replace(columns_map, inplace=True)

titles = set()

for index, row in train_data.iterrows():
	m = re.search(", ([^\\.]*)", row['Name'])
	title = m.group(1)
	titles.add(title)

for index, row in test_data.iterrows():
	m = re.search(", ([^\\.]*)", row['Name'])
	title = m.group(1)
	titles.add(title)

titles.add("C")
titles.add("Q")
titles.add("S")
new_cols = dict()
for title in titles:
	new_cols[title] = [0 for x in range(0, train_data.shape[0])]


for index, row in train_data.iterrows():
	new_cols[row['Embarked']][index - 1] = 1
	m = re.search(", ([^\\.]*)", row['Name'])
	title = m.group(1)
	new_cols[title][index - 1] = 1

for column in new_cols.keys():
	train_data[column] = new_cols[column]

new_cols = dict()
for title in titles:
	new_cols[title] = [0 for x in range(0, test_data.shape[0])]

for index, row in test_data.iterrows():
	new_cols[row['Embarked']][index - 1 - train_data.shape[0]] = 1
	m = re.search(", ([^\\.]*)", row['Name'])
	title = m.group(1)
	new_cols[title][index - 1 - train_data.shape[0]] = 1

for column in new_cols.keys():
	test_data[column] = new_cols[column]

both = train_data.append(test_data)

title_age_sum = dict()
title_count = dict()

for title in titles:
	title_age_sum[title] = 0
	title_count[title] = 0

for index, row in both.iterrows():
	m = re.search(", ([^\\.]*)", row['Name'])
	title = m.group(1)
	if np.isnan(row['Age']):
		continue
	title_count[title] += 1
	title_age_sum[title] += row['Age']

for index, row in test_data.iterrows():
	m = re.search(", ([^\\.]*)", row['Name'])
	title = m.group(1)
	if not np.isnan(row['Age']):
		continue
	test_data.at[index, 'Age'] = title_age_sum[title] / title_count[title]


for index, row in train_data.iterrows():
	m = re.search(", ([^\\.]*)", row['Name'])
	title = m.group(1)
	if not np.isnan(row['Age']):
		continue
	train_data.at[index, 'Age'] = title_age_sum[title] / title_count[title]

del both['Name']
del both['Cabin']
del both['Ticket']
del both['Embarked']
del both['Survived']

mn = both.min()
mx = both.max()

del test_data['Name']
del test_data['Cabin']
del test_data['Ticket']
del test_data['Embarked']

y_train = train_data.loc[:, 'Survived']

del train_data['Name']
del train_data['Cabin']
del train_data['Ticket']
del train_data['Embarked']
del train_data['Survived']

test_data=(test_data-mn)/(mn+mx)
train_data=(train_data-mn)/(mn+mx)

X_train = train_data.loc[:]

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=10)

truth = y_train.values


"""----Genetic Programming Setup----"""


pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 27), bool, "IN")

"""Need to find best primitives/terminals"""

# boolean operators
pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)
pset.addPrimitive(operator.xor, [bool, bool], bool)

# floating point operators
def safeDiv(left, right):
    try: return left / right
    except ZeroDivisionError: return 0
pset.addPrimitive(operator.add, [float,float], float)
pset.addPrimitive(operator.sub, [float,float], float)
pset.addPrimitive(operator.mul, [float,float], float)
pset.addPrimitive(safeDiv, [float,float], float)

# logic operators
def if_then_else(input, output1, output2):
    if input: return output1
    else: return output2

pset.addPrimitive(operator.lt, [float, float], bool)
pset.addPrimitive(operator.eq, [float, float], bool)
pset.addPrimitive(if_then_else, [bool, float, float], float)

# terminals
for var in "abcdefghij":
    pset.addEphemeralConstant(var, lambda: random.random() * 100, float)
pset.addTerminal(0, bool)
pset.addTerminal(1, bool)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

random.seed(25)


toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evaluate(individual,points,pset):
    func = gp.compile(expr=individual,pset=pset)
    predictions = [func(*points[x][:27]) for x in range(len(points))]
    tn, fp, fn, tp = confusion_matrix(truth, predictions).ravel()
    return fp/(fp+tp),fn/(fn+tn)
    
toolbox.register("evaluate", evaluate, points=X_train.values, pset=pset)

"""Need to find best selection, mating, mutation methods"""

toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("mut_eph", gp.mutEphemeral, mode="one")

"""Need to test different max heights"""

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("expr_mut", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def pareto_dominance(ind1, ind2):
    not_equal = False
    for value_1, value_2 in zip(ind1.fitness.values, ind2.fitness.values):
        if value_1 > value_2:
            return False
        elif value_1 < value_2:
            not_equal = True
    return not_equal


"""----Genetic Algorithm----"""

popSize = 400
mateRate = .8
mutRate = .3

def evolvePop(popSize,mateRate,mutRate):
    
    pop = toolbox.population(n=popSize)
    hof = tools.ParetoFront()
    
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    hof.update(pop)
    
    for g in range(50):
        print("-- Generation %i --" % g)
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < mateRate/100:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
    
        for mutant in offspring:
            if random.random() < mutRate/100:
                toolbox.mutate(mutant)
                del mutant.fitness.values
            if random.random() < mutRate/100:
                toolbox.mut_eph(mutant)
                del mutant.fitness.values
                
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        pop[:] = offspring
        hof.update(pop)
    return hof,pop

def graphPareto(hof,pop):
    fitness_1 = [ind.fitness.values[0] for ind in hof] # % FP
    fitness_2 = [ind.fitness.values[1] for ind in hof] # % FN
    pop_1 = [ind.fitness.values[0] for ind in pop]
    pop_2 = [ind.fitness.values[1] for ind in pop]
    
    plt.scatter(pop_1, pop_2, color='b')
    plt.scatter(fitness_1, fitness_2, color='r')
    plt.plot(fitness_1, fitness_2, color='r', drawstyle='steps-post')
    plt.xlabel("False Positives")
    plt.ylabel("False Negatives")
    plt.title("Pareto Front")
    plt.show()
    
    f1 = np.array(fitness_1)
    f2 = np.array(fitness_2)
    
    print("Area Under Curve: %s" % (np.sum(np.abs(np.diff(f1))*f2[:-1])))

hof, pop = evolvePop(popSize,mateRate,mutRate)
graphPareto(hof,pop)
    
#TREE VISUALIZATION CODE
pset.renameArguments(ARG0="x")
pset.renameArguments(ARG1="y")
expr = []
for n in hof:
    expr = expr + n

nodes, edges, labels = gp.graph(expr)
    
g = pgv.AGraph()
g.add_nodes_from(nodes)
g.add_edges_from(edges)
g.layout(prog="dot")
        
for i in nodes:
    n = g.get_node(i)
    n.attr["label"] = labels[i]
        
g.draw("tree.pdf")    


    
"""Random search stuff to optimize hyperparameters"""
    
#def getRandomSearches(numSearches, ranges):
#    """Int for how many tuples of random parameters you want, list of tuples of range of each parameter"""
#    paramList = []
#    for n in range(numSearches):
#        duplicate = True
#        #this loop goes forever if you are asking for more outputs than are possible with your ranges, so don't do that
#        while duplicate:
#            paramTup = ()
#            for r in ranges:
#                paramTup += (random.randint(r[0],r[1]),)
#            duplicate = paramTup in paramList
#        paramList.append(paramTup)
#    return paramList
#
#popRange = (1,500)
#mateRange = (0,100)
#mutRange = (0,100)
#
#numSearches = 100
#
#params = getRandomSearches(numSearches,[popRange,mateRange,mutRange])
#
#hofList = []
#popList = []
#
#count = 0
#for p in params:
#    count += 1
#    print('test ' + str(count))
#    hof,pop = evolvePop(p[0],p[1],p[2])
#    hofList.append(hof)
#    popList.append(pop)
#
#aucList = []
#for hof in hofList:
#    fitness_1 = [ind.fitness.values[0]/596 for ind in hof] # % FP
#    fitness_2 = [ind.fitness.values[1]/596 for ind in hof] # % FN 
#    f1 = np.array(fitness_1)
#    f2 = np.array(fitness_2)
#    aucList.append(np.sum(np.abs(np.diff(f1))*f2[:-1]))

            
# Tree Visualization

