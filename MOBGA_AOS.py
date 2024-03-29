from copy import deepcopy
from dataclasses import dataclass
import time
import numpy as np
from knn import knn
import matplotlib.pyplot as plt


@dataclass
class chromosome:
    genes: np.empty(0, dtype=bool)
    error: float = 0.0
    n_features: int = 0


class MOBGA_AOS:
    """
    MOBGA-AOS is a multi-objective optimization algorithm for feature
    selection in classification problems

    Objective 1: Classification error achieved by evaluating solutions
    using k-NN (k = 3) with n fold cross-validation (n = 3). Must be Minimized.

    Objective 2: Number of selected features. Must be Minimized.
    """

    def __init__(self, X, y, maxFEs=10000, popsize=100):
        # Id = current time
        self.id = time.strftime("%Y%m%d-%H%M%S")
        self.X = X
        self.y = y
        self.maxFEs = maxFEs
        self.popsize = popsize
        self.n_features = X.shape[1]
        self.n_samples = X.shape[0]
        self.n_classes = len(np.unique(y))
        self.n_fold = 3
        self.k = 5
        self.pc = 0.9
        self.pm = 1 / self.n_features
        self.Q = 5
        self.OSP = np.full(self.Q, 1 / self.Q)
        self.LP = 5
        self.RD = []
        self.row_RD = np.zeros(self.Q)
        self.PN = []
        self.row_PN = np.zeros(self.Q)
        self.population = []
        self.delta = 0.0001
        self.pareto_front = []
        self.crossover_operators = {
            0: self.single_point_crossover,
            1: self.two_point_crossover,
            2: self.uniform_crossover,
            3: self.shuffle_crossover,
            4: self.reduced_surrogate_crossover,
        }
        self.knn_model = knn(k=self.k, n_fold=self.n_fold)
        self.knn_model.fit(self.X, self.y)
        self.igd = []
        self.hv = []
        self.best_hv = 0
        self.same_hv = 0
        self.max_same_hv = 10
        self.p_true = 1 / self.n_features

        self.initialize_population()

    def initialize_population(self):
        """
        Initialize population with random solutions
        """
        for _ in range(self.popsize):
            # Initialize empty solution
            if self.n_features > 250:
                solution = np.random.choice(
                    [True, False],
                    size=self.n_features,
                    p=[self.p_true, 1 - self.p_true],
                )
            else:
                solution = np.random.choice([True, False], size=self.n_features)
            # Evaluate solution
            solution_error, solution_n_features = self.evaluate(solution)
            # Add solution to population
            self.population.append(
                chromosome(
                    genes=solution,
                    error=solution_error,
                    n_features=solution_n_features,
                )
            )
        self.population = np.array(self.population)

    def evaluate(self, solution):
        """
        Return classification error and number of features of solution
        """
        return 1 - self.knn_model.accuracy(solution), np.sum(solution)

    def single_point_crossover(self, parent1: chromosome, parent2: chromosome) -> tuple:
        """
        Single point crossover
        """
        # Get crossover point
        crossover_point = np.random.randint(self.n_features)
        # Perform crossover
        child1 = np.concatenate(
            (parent1.genes[:crossover_point], parent2.genes[crossover_point:])
        )
        child2 = np.concatenate(
            (parent2.genes[:crossover_point], parent1.genes[crossover_point:])
        )
        # Evaluate children
        child1_error, child1_n_features = self.evaluate(child1)
        child2_error, child2_n_features = self.evaluate(child2)
        # Return children
        return (
            chromosome(genes=child1, error=child1_error, n_features=child1_n_features),
            chromosome(genes=child2, error=child2_error, n_features=child2_n_features),
        )

    def two_point_crossover(self, parent1: chromosome, parent2: chromosome) -> tuple:
        """
        Two point crossover
        """
        # Get crossover points
        crossover_points = np.random.choice(
            np.arange(1, self.n_features), size=2, replace=False
        )
        crossover_points.sort()
        # Perform crossover
        child1 = np.concatenate(
            (
                parent1.genes[: crossover_points[0]],
                parent2.genes[crossover_points[0] : crossover_points[1]],
                parent1.genes[crossover_points[1] :],
            )
        )
        child2 = np.concatenate(
            (
                parent2.genes[: crossover_points[0]],
                parent1.genes[crossover_points[0] : crossover_points[1]],
                parent2.genes[crossover_points[1] :],
            )
        )
        # Evaluate children
        child1_error, child1_n_features = self.evaluate(child1)
        child2_error, child2_n_features = self.evaluate(child2)
        # Return children
        return (
            chromosome(genes=child1, error=child1_error, n_features=child1_n_features),
            chromosome(genes=child2, error=child2_error, n_features=child2_n_features),
        )

    def uniform_crossover(self, parent1: chromosome, parent2: chromosome) -> tuple:
        """
        Uniform crossover
        """
        # Get crossover mask
        crossover_mask = np.random.choice([True, False], size=self.n_features)
        # Perform crossover
        child1 = np.where(crossover_mask, parent1.genes, parent2.genes)
        child2 = np.where(crossover_mask, parent2.genes, parent1.genes)
        # Evaluate children
        child1_error, child1_n_features = self.evaluate(child1)
        child2_error, child2_n_features = self.evaluate(child2)
        # Return children
        return (
            chromosome(genes=child1, error=child1_error, n_features=child1_n_features),
            chromosome(genes=child2, error=child2_error, n_features=child2_n_features),
        )

    def shuffle_crossover(self, parent1: chromosome, parent2: chromosome) -> tuple:
        """
        Randomly shuffle genes of parents, then apply
        single-point crossover on shuffled parents.
        """
        # Deep copy parents
        parent1 = deepcopy(parent1)
        parent2 = deepcopy(parent2)
        # Shuffle genes of parents
        np.random.shuffle(parent1.genes)
        np.random.shuffle(parent2.genes)
        # Apply single-point crossover
        return self.single_point_crossover(parent1, parent2)

    def reduced_surrogate_crossover(self, parent1: chromosome, parent2: chromosome):
        """
        Only allow crossover points where parents have different genes.
        Randomly select one such point and apply single-point crossover.
        """
        # Get crossover points where parents have different genes
        crossover_points = np.where(parent1.genes != parent2.genes)[0]
        # If there are no such points, return parents
        if len(crossover_points) == 0:
            return parent1, parent2
        # Randomly select crossover point
        crossover_point = np.random.choice(crossover_points)
        # Perform crossover
        child1 = np.concatenate(
            (parent1.genes[:crossover_point], parent2.genes[crossover_point:])
        )
        child2 = np.concatenate(
            (parent2.genes[:crossover_point], parent1.genes[crossover_point:])
        )
        # Evaluate children
        child1_error, child1_n_features = self.evaluate(child1)
        child2_error, child2_n_features = self.evaluate(child2)
        # Return children
        return (
            chromosome(genes=child1, error=child1_error, n_features=child1_n_features),
            chromosome(genes=child2, error=child2_error, n_features=child2_n_features),
        )

    def mutation(self, offspring: chromosome) -> chromosome:
        """
        Uniform mutation with rate of Pm.
        """
        # Get mutation mask
        mutation_mask = np.random.choice(
            [True, False], size=self.n_features, p=[self.pm, 1 - self.pm]
        )
        # Perform mutation
        offspring.genes = np.where(mutation_mask, ~offspring.genes, offspring.genes)
        # Evaluate offspring
        offspring.error, offspring.n_features = self.evaluate(offspring.genes)
        # Return offspring
        return offspring

    def dominate(self, solution1: chromosome, solution2: chromosome) -> bool:
        """
        Check if solution1 dominates solution2.
        """
        # If solution1 is better than solution2 in both objectives
        if (
            solution1.error <= solution2.error
            and solution1.n_features <= solution2.n_features
        ):
            # If solution1 is better than solution2 in at least one objective
            if (
                solution1.error < solution2.error
                or solution1.n_features < solution2.n_features
            ):
                # Solution1 dominates solution2
                return True
        # Solution1 does not dominate solution2
        return False

    def fast_nondominated_sort(self, population, first_front_only=False):
        """
        Fast nondominated sort
        """
        # Initialize fronts
        fronts = []
        # Initialize front 1
        fronts.append([])
        # Initialize empty set of solutions that solution1 dominates
        S = [[] for _ in range(len(self.population))]
        # Initialize empty set of solutions that dominate solution1
        n = np.zeros(len(self.population), dtype=int)
        # For each solution in population
        for p in range(len(self.population)):
            # For each other solution in population
            for q in range(p, len(self.population)):
                # If solution1 dominates solution2
                if self.dominate(population[p], population[q]):
                    # Add solution2 to set of solutions that solution1 dominates
                    S[p].append(q)
                    n[q] += 1
                # If solution2 dominates solution1
                elif self.dominate(population[q], population[p]):
                    # Increment domination counter of solution1
                    S[q].append(p)
                    n[p] += 1
            # If solution1 is not dominated by any other solution
            if n[p] == 0:
                # Add solution1 to front 1
                fronts[0].append(p)

        # If only the first front is needed
        if first_front_only:
            # Return first front
            return fronts[0]

        # Initialize front counter
        i = 0
        # While front i is not empty
        while len(fronts[i]) > 0:
            # Initialize empty front i + 1
            fronts.append([])
            # For each solution in front i
            for p in fronts[i]:
                # For each solution that solution p dominates
                for q in S[p]:
                    # Decrement domination counter of solution q
                    n[q] -= 1
                    # If solution q is not dominated by any other solution
                    if n[q] == 0:
                        # Add solution q to front i + 1
                        fronts[i + 1].append(q)
            # Increment front counter
            i += 1
        del fronts[-1]
        # Return fronts
        return fronts

    def crowding_distance(self, front):
        """
        Crowding distance
        """
        # Get number of solutions in front
        n = len(front)
        # Initialize crowding distance
        distance = np.zeros(n)
        # For error and number of features
        for m in range(2):
            # Get objective values of solutions in front
            f = np.array(
                [getattr(front[i], ["error", "n_features"][m]) for i in range(n)]
            )
            # Get indices of sorted objective values
            sorted_indices = np.argsort(f)
            # Set boundary solutions' crowding distance to infinity
            distance[sorted_indices[[0, -1]]] = np.inf
            # Get objective values of boundary solutions
            f_min = f[sorted_indices[0]]
            f_max = f[sorted_indices[-1]]
            # For each solution in front
            for i in range(1, n - 1):
                # Add crowding distance
                distance[sorted_indices[i]] += (
                    f[sorted_indices[i + 1]] - f[sorted_indices[i - 1]]
                ) / (f_max - f_min + self.delta)

        # Return crowding distance
        return distance

    def selection(self):
        """
        Use NSGA-II's original fast nondominated sorting and crowding distance to select survivors.
        """
        # Get fronts
        fronts = self.fast_nondominated_sort(self.population)
        # Initialize empty set of survivors
        survivors = []
        # Initialize front counter
        i = 0
        # While adding front i to survivors does not exceed population size
        while len(survivors) + len(fronts[i]) < self.popsize:
            # Add front i to survivors
            survivors.extend(fronts[i])
            # Increment front counter
            i += 1
        # Sort remaining solutions in front i by crowding distance
        sorted_indices = np.argsort(self.crowding_distance(self.population[fronts[i]]))
        # Add solutions with largest crowding distance to survivors
        survivors.extend(
            [fronts[i][j] for j in sorted_indices[::-1]][
                : self.popsize - len(survivors)
            ]
        )
        # Return survivors
        return survivors

    def roulette_wheel_selection(self):
        """
        Select selection operator based on OSP"""
        return np.random.choice(self.Q, p=self.OSP)

    def credit_assignment(
        self,
        parent1: chromosome,
        parent2: chromosome,
        child1: chromosome,
        child2: chromosome,
        operator_idx: int,
    ):
        """
        Credit assignment
        """
        # Parent1 dominates the parent2
        if self.dominate(parent1, parent2):
            # If the first child do not dominated by parent
            if not self.dominate(parent1, child1):
                # Update RD
                self.row_RD[operator_idx] += 1
            else:
                # Update PN
                self.row_PN[operator_idx] += 1
            # The second child
            if not self.dominate(parent1, child2):
                # Update RD
                self.row_RD[operator_idx] += 1
            else:
                # Update PN
                self.row_PN[operator_idx] += 1
        # Parent2 dominates the parent1
        elif self.dominate(parent2, parent1):
            # If the first child do not dominated by parent
            if not self.dominate(parent2, child1):
                # Update RD
                self.row_RD[operator_idx] += 1
            else:
                # Update PN
                self.row_PN[operator_idx] += 1
            # The second child
            if not self.dominate(parent2, child2):
                # Update RD
                self.row_RD[operator_idx] += 1
            else:
                # Update PN
                self.row_PN[operator_idx] += 1
        # Parent1 and parent2 are non-dominated
        else:
            # If the first child is not dominated by the two parents
            if not (self.dominate(parent1, child1) or self.dominate(parent2, child1)):
                # Update RD
                self.row_RD[operator_idx] += 1
            else:
                # Update PN
                self.row_PN[operator_idx] += 1
            # The second child
            if not (self.dominate(parent1, child2) or self.dominate(parent2, child2)):
                # Update RD
                self.row_RD[operator_idx] += 1
            else:
                # Update PN
                self.row_PN[operator_idx] += 1

    def update_OSP(self):
        new_OSP = np.zeros(self.Q)
        for q in range(self.Q):
            # sum the qth column of RD and PN
            s1 = np.sum(np.array(self.RD)[:, q])
            s2 = np.sum(np.array(self.PN)[:, q])
            # OSP for the qth operator can be calculated as
            # to prevent being divided by zero
            s3 = self.delta if s1 == 0 else s1
            # refers to the probability assigned to qth operator
            s4 = s1 / (s3 + s2)
            # update the qth element of OSP
            new_OSP[q] = s4
        # normalize OSP
        self.OSP = new_OSP / np.sum(new_OSP)
        # Clear RD and PN
        self.RD = []
        self.row_RD = np.zeros(self.Q)
        self.PN = []
        self.row_PN = np.zeros(self.Q)

    def union(
        self, population: np.ndarray[chromosome], offspring: list[chromosome]
    ) -> list[chromosome]:
        """
        Union of population and offspring
        """
        set1 = set([tuple(solution.genes) for solution in population])
        set2 = set([tuple(solution.genes) for solution in offspring])
        union = set1.union(set2)
        population = []
        for solution in union:
            error, n_features = self.evaluate(np.array(solution))
            population.append(
                chromosome(
                    genes=np.array(solution),
                    error=error,
                    n_features=n_features,
                )
            )

        return np.array(population)

    def plot_pareto_front(self) -> None:
        """
        Plot Pareto front
        """

        # Get objective values of solutions in Pareto front
        f1 = np.array([self.population[i].n_features for i in self.pareto_front])
        f2 = np.array([self.population[i].error for i in self.pareto_front])
        # Plot all solutions
        plt.figure(figsize=(8, 6))
        plt.scatter(
            [solution.n_features for solution in self.population],
            [solution.error for solution in self.population],
            c="r",
            marker="o",
            s=30,
            label="All solutions",
        )
        # Plot Pareto front and connect them with lines
        # Sort solutions in Pareto front by number of features
        sorted_indices = np.argsort(f1)
        # Plot Pareto front
        plt.plot(f1[sorted_indices], f2[sorted_indices], c="b", label="Pareto front")
        plt.xlabel("Solution size")
        plt.ylabel("Classification error")
        plt.title(f"Pareto front {self.n_features}")
        # Save on Images folder
        plt.savefig(f"Images/{self.n_features}_{self.id}_pareto_front.png")
        plt.show()

    def calculate_hypervolume(self) -> float:
        """
        Calculate hypervolume
        """
        # Get reference point
        refrence_point = np.array([self.n_features, 1])
        # Get objective values of solutions in Pareto front
        f1 = np.array([self.population[i].n_features for i in self.pareto_front])
        f2 = np.array([self.population[i].error for i in self.pareto_front])
        sorted_indices = np.argsort(f1)
        # Initialize hypervolume
        hypervolume = (
            (refrence_point[0] - f1[sorted_indices[0]])
            * (refrence_point[1] - f2[sorted_indices[0]])
            / self.n_features
        )
        # For each solution in Pareto front
        for i in range(1, len(self.pareto_front)):
            # Add hypervolume
            hypervolume += (
                (refrence_point[0] - f1[sorted_indices[i]])
                * (f2[sorted_indices[i - 1]] - f2[sorted_indices[i]])
                / self.n_features
            )
        # Return hypervolume
        return hypervolume

    def plot_hypervolume(self) -> None:
        """
        Plot hypervolume
        """
        plt.figure(figsize=(8, 6))
        plt.plot(self.hv)
        plt.xlabel("Generation")
        plt.ylabel("Hypervolume")
        plt.title(f"Hypervolume {self.n_features}")
        # Save on Images folder
        plt.savefig(f"Images/{self.n_features}_{self.id}_hypervolume.png")
        plt.show()

    def run(self):
        """
        Run MOBGA-AOS
        """
        # Counter for updating OSP each LP generation
        k = 0
        # For each generation
        nEF = 0
        while nEF < self.maxFEs and self.same_hv < self.max_same_hv:
            # Initialize empty set of offspring
            print(f"Iteration {int(nEF/self.popsize)}, same_hv: {self.same_hv}")
            offspring = []
            # For each parent
            for i in range(self.popsize // 2):
                operator_idx = self.roulette_wheel_selection()
                # Randomly select two individuals as parents
                id1, id2 = np.random.choice(self.popsize, size=2, replace=False)
                parent1 = self.population[id1]
                parent2 = self.population[id2]
                # Perform crossover
                child1, child2 = self.crossover_operators[operator_idx](
                    parent1, parent2
                )
                # Perform mutation
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                # creditAssignment
                self.credit_assignment(parent1, parent2, child1, child2, operator_idx)
                # Add offspring to set of offspring
                offspring.append(child1)
                offspring.append(child2)
            k += 1
            # Append nReward to the kth row of RD
            self.RD.append(self.row_RD)
            # Append nPenalty to the kth row of PN
            self.PN.append(self.row_PN)
            # Update OSP
            if k == self.LP:
                self.update_OSP()
                k = 0
            # R ← P union Pnew
            self.population = self.union(self.population, offspring)
            # Environmental Selection
            survivors = self.selection()
            # P ← R
            self.population = self.population[survivors]
            # Select non-dominated solutions in P as PF
            self.pareto_front = self.fast_nondominated_sort(self.population, True)
            # Update nEF
            nEF += self.popsize
            list(self.population[self.pareto_front])
            # Store IGD and HV of current generation
            hv = self.calculate_hypervolume()
            self.hv.append(hv)
            if hv > self.best_hv + self.delta * 10:
                self.best_hv = hv
                self.same_hv = 0
            else:
                self.same_hv += 1
            print("HV: ", hv)
        # Return Pareto front and maximum hypervolume
        return self.pareto_front, self.best_hv
