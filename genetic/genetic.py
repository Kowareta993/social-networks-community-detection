import copy
import networkx as nx
import random
import sys
import matplotlib.pyplot as plt


class Chromosome:
    def __init__(self, data):
        self.data = copy.deepcopy(data)

    def __repr__(self):
        return f"[{','.join([str(d) for d in self.data])}]"


class Problem:
    _problem = None

    def __new__(cls, G):
        if cls._problem is None:
            cls._problem = super(Problem, cls).__new__(cls)
            cls._problem.G = G
            cls._problem.nodes = list(G.nodes)
            cls._problem.matrix = nx.to_numpy_array(G, nodelist=G.nodes)
        return cls._problem

    def initial(self, size):
        population = []
        for i in range(size):
            data = []
            for j in range(len(self.nodes)):
                rnd = random.randint(0, len(self.nodes) - 1)
                while self.matrix[rnd, j] != 1:
                    rnd = random.randint(0, len(self.nodes) - 1)
                data.append(rnd)
            population.append(Chromosome(data))
        return population

    def connected_components(self, chromosome):
        graph = [[] for _ in range(len(chromosome.data))]
        for i in range(len(self.nodes)):
            graph[i].append(chromosome.data[i])
            graph[chromosome.data[i]].append(i)
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

    def eval(self, chromosome):
        m = 2.0 * (len(self.G.edges))
        n = len(self.G.nodes)
        nodes = list(self.G.nodes)
        groups = self.connected_components(chromosome)
        return sum(
            [self.matrix[i, j] - self.G.degree[nodes[i]] * self.G.degree[nodes[j]] / m
             for i in range(n) for j in range(n) if groups[i] == groups[j] and i != j]) / m

    def crossover(self, parents):
        children = []
        for i in range(int(len(parents) / 2)):
            c1 = parents[2 * i]
            c2 = parents[2 * i + 1]
            mask = random.choices([0, 1], k=len(self.nodes))
            child1 = []
            child2 = []
            for i in range(len(self.nodes)):
                if mask[i]:
                    child1.append(c1.data[i])
                    child2.append(c2.data[i])
                else:
                    child1.append(c2.data[i])
                    child2.append(c1.data[i])
            children += [Chromosome(child1), Chromosome(child2)]
        return children

    def mutate(self, c):
        place = random.choice(range(len(c.data)))
        if self.G.degree(self.nodes[place]) > 0:
            rnd = random.randint(0, len(self.nodes) - 1)
            while self.matrix[place, rnd] != 1:
                rnd = random.randint(0, len(self.nodes) - 1)
            c.data[place] = rnd
        else:
            c.data[place] = place


class Genetic:
    def __init__(self, problem):
        self.size = 100
        self.mutation_p = 0.9
        self.problem = problem
        self.population = None
        self.fitness_scores = None

    def initial(self):
        return problem.initial(self.size)

    def fitness(self, chromosome):
        return self.problem.eval(chromosome)

    def crossover(self, parents):
        return problem.crossover(parents)

    def mutate(self, chromosome):
        if random.choices([0, 1], weights=[1 - self.mutation_p, self.mutation_p], k=1)[0] == 1:
            problem.mutate(chromosome)

    def terminate(self, n):
        self.population = [x for _, x in sorted(zip(self.fitness_scores, self.population), key=lambda x: x[0])]
        self.fitness_scores = sorted(self.fitness_scores)
        del self.population[:n]
        del self.fitness_scores[:n]

    def run(self, steps):
        self.population = self.initial()
        self.fitness_scores = [self.fitness(chromosome) for chromosome in self.population]
        for step in range(steps):
            parents = [self.population[i] for i in
                       random.choices(range(len(self.population)), weights=self.fitness_scores, k=int(self.size / 2))]

            children = self.crossover(parents)
            for child in children:
                self.mutate(child)
                self.population.append(child)
                self.fitness_scores.append(self.fitness(child))
            self.terminate(len(self.population) - self.size)


if __name__ == "__main__":
    G = nx.read_edgelist(sys.argv[1], create_using=nx.Graph(), nodetype=int)
    problem = Problem(G)
    algorithm = Genetic(problem)
    algorithm.run(1000)
    print(algorithm.population[0])
    print(algorithm.fitness_scores[0])
    labels = problem.connected_components(algorithm.population[0])
    G.remove_edges_from(nx.selfloop_edges(G))
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'),
                           node_color=labels, node_size=200)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges)
    plt.show()
