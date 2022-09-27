import networkx as nx
import numpy as np
import random
import copy
import sys
import math
import matplotlib.pyplot as plt

problem = None

class Nest:
    def __init__(self, n, k):
        self.eggs = [Egg(n) for i in range(k)]
        self.score = None
    
    def fitness_score(self, fitness):
        self.score = sum([egg.fitness_score(fitness) for egg in self.eggs]) / len(self.eggs)
        return self.score
    
    def flight(self, steps):
        for egg in self.eggs:
            egg.flight(steps)

    def best_egg(self):
        best = self.eggs[0]
        for egg in self.eggs:
            if egg.score > best.score:
                best = egg
        return best

class Egg:
    def __init__(self, n):
        self.solution = problem.random_solution()
        self.score = None

    def fitness_score(self, fitness):
        self.score = fitness(self.solution)
        return self.score
    
    def flight(self, steps):
        self.solution = problem.flight(self.solution, steps)

class CuckooSearch:
    def __init__(self, nests, fitness):
        self.nests = nests
        self.fitness = fitness
        self.size = len(nests)
        self.dimension = len(nests[0].eggs[0].solution)
        self.Lambda = 1.2
        self.pa = 0.1
        self.eggs = 1
        
    def levi_flight(self, Lambda, size):
        #generate step from levy distribution
        sigma1 = np.power((math.gamma(1 + Lambda) * np.sin((np.pi * Lambda) / 2)) \
                        / math.gamma((1 + Lambda) / 2) * np.power(2, (Lambda - 1) / 2), 1 / Lambda)
        sigma2 = 1
        u = np.random.normal(0, sigma1, size=size)
        v = np.random.normal(0, sigma2, size=size)
        step = u / np.power(np.fabs(v), 1 / Lambda)

        return step    # return np.array (ex. [ 1.37861233 -1.49481199  1.38124823])

    def run(self, steps):
        [nest.fitness_score(self.fitness) for nest in self.nests]
        for step in range(steps):
            news = []
            for nest in self.nests:
                new_nest = copy.deepcopy(nest)
                new_nest.flight(self.levi_flight(self.Lambda, self.dimension))
                new_nest.fitness_score(self.fitness)
                news.append(new_nest)
            self.nests += news
            self.nests = sorted(self.nests, key=lambda nest: nest.score, reverse=True) 
            self.nests = self.nests[:int(self.size*(1-self.pa))]
            for i in range(int(self.size*self.pa)):
                new_nest = Nest(self.dimension, random.randint(1, self.eggs))
                new_nest.fitness_score(self.fitness)
                self.nests.append(new_nest)
        self.nests = sorted(self.nests, key=lambda nest: nest.score, reverse=True) 

class Problem:
    _problem = None

    def __new__(cls, G):
        if cls._problem is None:
            cls._problem = super(Problem, cls).__new__(cls)
            cls._problem.G = G
            cls._problem.nodes = list(G.nodes)
            cls._problem.matrix = nx.to_numpy_array(G, nodelist=G.nodes)
        return cls._problem

    def initial(self, n, k):
        return [Nest(len(self.nodes), k) for i in range(n)]

    def random_solution(self):
        solution = []
        for i in range(len(self.nodes)):
            rnd = random.randint(0, len(self.nodes) - 1)
            while self.matrix[i, rnd] != 1:
                rnd = random.randint(0, len(self.nodes) - 1)
            solution.append(rnd)
        return solution
            
    def flight(self, solution, steps):
        for i in range(len(steps)):
            neigbors = list(self.G.neighbors(self.nodes[i]))
            idx = neigbors.index(self.nodes[solution[i]])
            solution[i] = self.nodes.index(neigbors[(int(steps[i]) + idx) % len(neigbors)])
        return solution

    def connected_components(self, solution):
        graph = [[] for _ in range(len(solution))]
        for i in range(len(self.nodes)):
            graph[i].append(solution[i])
            graph[solution[i]].append(i)
        labels = [0 for _ in range(len(graph))]
        i = 0
        k = 1
        while i != len(graph):
            if labels[i] != 0:
                i += 1
                continue
            labels[i] = k
            queue = [i]
            while len(queue):
                n = queue.pop()
                for j in range(len(graph[n])):
                    if labels[graph[n][j]] == 0:
                        queue.append(graph[n][j])
                        labels[graph[n][j]] = labels[i]
            i += 1
            k += 1
        return labels

    def eval(self, solution):
        m = 2.0 * (len(self.G.edges))
        n = len(self.G.nodes)
        nodes = list(self.G.nodes)
        groups = self.connected_components(solution)
        return sum(
            [self.matrix[i, j] - self.G.degree[nodes[i]] * self.G.degree[nodes[j]] / m
             for i in range(n) for j in range(n) if groups[i] == groups[j] and i != j]) / m


if __name__ == "__main__":
    n = 20
    G = nx.read_edgelist(sys.argv[1], create_using=nx.Graph(), nodetype=int)
    problem = Problem(G)
    nests = problem.initial(n, 1)
    algorithm = CuckooSearch(nests, problem.eval)
    algorithm.run(1000)
    print(algorithm.nests[0].best_egg().solution)
    print(algorithm.nests[0].best_egg().score)
    labels = problem.connected_components(algorithm.nests[0].best_egg().solution)
    pos = nx.nx_agraph.graphviz_layout(G, prog="neato")
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'),
                           node_color=labels, node_size=200)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges)
    plt.show()

