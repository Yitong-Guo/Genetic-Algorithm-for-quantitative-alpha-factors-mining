import operator
import time
import random
from copy import deepcopy
import importlib
import inspect
import numpy as np
import pandas as pd
from datetime import datetime

from deap import gp, creator, base, tools, algorithms

from utils import *
from terminal_process import TerminalProcess
from evaluate_func import *


class GPProcess(TerminalProcess):
    """
    这里是遗传过程程序
    """
    def __init__(self,
                 train_data,
                 val_data,
                 test_data,
                 train_evaluation,
                 valid_evaluation,
                 population_num=1000,
                 arity=0,
                 batch_size=50,
                 generation=5,
                 initial_depth=2,
                 max_depth=3,
                 random_seed=1,
                 cals='all'):
        """
        :param train_data: the data used in genetic process
        :param val_data: the data to select the qualified formula among formulas generated after genetic process
        :param test_data: the data you want to generate its factor
        :param train_evaluation: The goal in training process (evaluation fuction)
        :param valid_evaluation: The goal in validation process
        :param population_num: all the number of formula trees in the population
        :param arity: arity
        :param batch_size: the number of formulas involved in each group
        :param generation: number of evolutionary generations
        :param initial_depth: The maximum depth of the initial generated formula tree
        :param max_depth: control the maximum depth during the iteration process
        :param random_seed: set the random state of genetic programming
        :param cals: The selected operators, either the string 'all' or a list containing the operators name
        """
        super(GPProcess, self).__init__(train_data, val_data, test_data)

        self._population_num = population_num
        self.batch_size = batch_size
        self.generation = generation
        self.initial_depth = initial_depth
        self.max_depth = max_depth
        self.random_seed = random_seed
        self.train_evaluation = train_evaluation
        self.valid_evaluation = valid_evaluation
        self.cals = cals
        if isinstance(cals, str):
            if cals != 'all':
                raise ValueError("When you use string, cals must be 'all', or you can use list to choose")
        else:
            if not isinstance(cals, list):
                raise ValueError("Cals supposed to be a list when you want to choose, or use 'all' to choose all")

        self._arity = arity

        self.pset_train = self.init_primitive_set('train')
        self.pset_valid = self.init_primitive_set('valid')
        self.pset_test = self.init_primitive_set('test')

        self.toolbox_multi = self.init_toolbox_multi()
        self.toolbox_single1 = self.init_toolbox_single(target_index=0)
        self.toolbox_single2 = self.init_toolbox_single(target_index=1)

    def evalfunc(self, individual):
        """the evaluation function for train set"""
        # code = str(individual)
        # print(individual)
        new_factor = self.toolbox_multi.compile(expr=individual)
        y = self.returns_train.clone()
        evaluation_result = evaluate(new_factor, y, self.train_evaluation)

        return evaluation_result[0], evaluation_result[1]  # 返回两个适应度值

    def evalfunc_valid(self, individual):
        """the evaluation function for valid set"""
        # 评价目标：rankIC
        new_factor = self.toolbox_multi.compile_valid(expr=individual)
        y = self.returns_val.clone()
        evaluation_result = evaluate(new_factor, y, self.valid_evaluation)
        if len(evaluation_result) == 1:
            return evaluation_result[0]
        elif len(evaluation_result) == 2:
            return evaluation_result[0], evaluation_result[1]

    def add_elementary_cals(self, pset):
        """add elementary operators"""
        module1 = importlib.import_module('cal.cal_elementary')
        elementary_funcs = {name: func for name, func in inspect.getmembers(module1, inspect.isfunction)}

        for name, func in elementary_funcs.items():
            if self.cals != 'all':
                if name not in self.cals:
                    continue

            if name in ['log_torch', 'exp_torch', 'sqrt_torch', 'square_torch', 'sin_torch', 'cos_torch',
                        'neg', 'inv', 'sign_torch', 'abs_torch', 'sigmoid_torch', 'haedsigmoid_torch', 'gelu_torch']:
                pset.addPrimitive(func, 1)
            elif name in ['add_const', 'mul_const']:
                for const in range(1, 11):
                    pset.addPrimitive(make_partial_func(func, constant=const), 1)
            elif name in ['add', 'sub', 'mul', 'div']:
                pset.addPrimitive(func, 2)
            elif name == 'leakyrelu_torch':
                for alpha in np.arange(0.1, 1.0, 0.1):
                    pset.addPrimitive(make_partial_func(func, alpha=alpha), 1)

        return pset

    def add_cross_sectional_cals(self, pset):
        """add cross sectional operators"""
        module2 = importlib.import_module("cal.cal_cross_sectional")
        cross_sectional_funcs = {name: func for name, func in inspect.getmembers(module2, inspect.isfunction)}

        for name, func in cross_sectional_funcs.items():
            if self.cals != 'all':
                if name not in self.cals:
                    continue

            if name == 'rank_pct_torch':
                pset.addPrimitive(func, 1)
            elif name in ['xs_cutquartile_torch', 'xs_cutzscore_torch']:
                for alpha in np.arange(1.5, 3.0, 0.5):
                    pset.addPrimitive(make_partial_func(func, alpha=alpha), 1)
            elif name == 'xs_regres_torch':
                pset.addPrimitive(func, 2)
            elif name == 'xs_sortreverse_torch':
                for rank in range(1, 11):
                    for mode in [0, 1, 2]:
                        pset.addPrimitive(make_partial_func(func, rank=rank, mode=mode), 1)
            elif name == 'xs_zscorereverse_torch':
                for alpha in np.arange(1.5, 3.0, 0.5):
                    for mode in [0, 1, 2]:
                        pset.addPrimitive(make_partial_func(func, alpha=alpha, mode=mode), 1)
            elif name == 'xs_grouping_sortreverse_torch':
                for rank in range(1, 9):
                    for mode in [0, 1, 2]:
                        pset.addPrimitive(make_partial_func(func, rank=rank, mode=mode), 2)
            elif name == 'xs_grouping_zscorereverse_torch':
                for alpha in np.arange(1.5, 3.0, 0.5):
                    for mode in [0, 1, 2]:
                        pset.addPrimitive(make_partial_func(func, alpha=alpha, mode=mode), 2)

        return pset

    def add_time_series_cals(self, pset):
        """add time series operators"""
        module3 = importlib.import_module('cal.cal_time_series')
        time_series_funcs = {name: func for name, func in inspect.getmembers(module3, inspect.isfunction)}

        for name, func in time_series_funcs.items():
            if self.cals != 'all':
                if name not in self.cals:
                    continue

            if name in ['ts_delay', 'ts_delta', 'ts_pctchange']:
                for window in range(1, 11):
                    pset.addPrimitive(make_partial_func(func, window=window), 1)
            elif name in ['ts_max', 'ts_min', 'ts_argmax', 'ts_argmin', 'ts_sum', 'ts_mean']:
                for window in range(2, 21):
                    pset.addPrimitive(make_partial_func(func, window=window), 1)
            elif name in ['ts_weighted_mean']:
                for window in range(2, 21):
                    pset.addPrimitive(make_partial_func(func, window=window), 2)
            elif name in ['ts_roll_rank', 'ts_roll_zscore', 'ts_fourier_peak']:
                for window in range(10, 61, 5):
                    pset.addPrimitive(make_partial_func(func, window=window), 1)
            elif name in ['ts_correlation', 'ts_covariance', 'ts_regbeta', 'ts_regalpha', 'ts_regres']:
                for window in range(10, 61, 5):
                    pset.addPrimitive(make_partial_func(func, window=window), 2)
            elif name in ['ts_median', 'ts_var', 'ts_stddev', 'ts_avgdev', 'ts_kurt', 'ts_skew']:
                for window in range(5, 21):
                    pset.addPrimitive(make_partial_func(func, window=window), 1)

            elif name in ['ts_asc_sort_cut', 'ts_dec_sort_cut']:
                for window in range(25, 56, 5):
                    for n in range(1, 9, 2):
                        for mode in range(1, 3):
                            pset.addPrimitive(make_partial_func(func, window=window, n=n, mode=mode), 1)
            elif name in ['ts_asc_zscore_cut', 'ts_dec_zscore_cut']:
                for window in range(25, 56, 5):
                    for a in [0.5, 1.0, 1.5, 2.0]:
                        for mode in range(1, 3):
                            pset.addPrimitive(make_partial_func(func, window=window, a=a, mode=mode), 1)
            elif name in ['ts_group_asc_sort_cut', 'ts_group_dec_sort_cut']:
                for window in range(25, 56, 5):
                    for n in range(1, 9, 2):
                        for mode in range(1, 3):
                            pset.addPrimitive(make_partial_func(func, window=window, n=n, mode=mode), 2)
            elif name in ['ts_group_asc_zscore_cut', 'ts_group_dec_zscore_cut']:
                for window in range(25, 56, 5):
                    for a in [0.5, 1.0, 1.5, 2.0]:
                        for mode in range(1, 3):
                            pset.addPrimitive(make_partial_func(func, window=window, a=a, mode=mode), 2)

        return pset

    def init_primitive_set(self, flag):
        """initiate the pset which contain operators and terminals"""
        pset = gp.PrimitiveSet("MAIN", self._arity)

        # add operators
        pset = self.add_elementary_cals(pset=pset)
        pset = self.add_cross_sectional_cals(pset=pset)
        pset = self.add_time_series_cals(pset=pset)

        # add terminals
        if flag == 'train':
            for name in self.train_attr:
                if name != 'returns':
                    pset.addTerminal(getattr(self, name + '_train'), name)

        elif flag == 'valid':
            for name in self.val_attr:
                if name != 'returns':
                    pset.addTerminal(getattr(self, name + '_val'), name)

        elif flag == 'test':
            for name in self.test_attr:
                if name != 'returns':
                    pset.addTerminal(getattr(self, name + '_test'), name)

        else:
            raise ValueError("flag should be 'train', 'valid', or 'test' ")

        return pset

    def init_toolbox_multi(self):
        """initiate the toolbox for multi target genetic programming"""
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset_train, min_=1, max_=self.initial_depth)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=self.pset_train)
        toolbox.register("compile_valid", gp.compile, pset=self.pset_valid)
        toolbox.register("compile_test", gp.compile, pset=self.pset_test)

        toolbox.register("evaluate", self.evalfunc)
        toolbox.register("evaluate_valid", self.evalfunc_valid)
        toolbox.register("select", tools.selNSGA2)

        return toolbox

    def init_toolbox_single(self, target_index):
        """initiate the toolbox for single target optimization"""
        creator.create(f"FitnessSingle{target_index}", base.Fitness, weights=(1.0,))
        creator.create(f"IndividualSingle{target_index}", gp.PrimitiveTree, fitness=getattr(creator, f"FitnessSingle{target_index}"))

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset_train, min_=1, max_=self.initial_depth)
        toolbox.register("individual", tools.initIterate, getattr(creator, f"IndividualSingle{target_index}"), toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=self.pset_train)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=self.pset_train)

        toolbox.register("evaluate", lambda ind: (self.evalfunc(ind)[target_index],))
        toolbox.register("select", tools.selBest)
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_depth))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_depth))

        return toolbox

    def convert_to_single(self, population, target_index):
        """
        Transforming multi-objective individuals into single objective individuals:
        mainly by separately extracting the two fitness indicators of the individual
        """
        single_population = []
        for ind in population:
            single_ind = creator.IndividualSingle0(ind) if target_index == 0 else creator.IndividualSingle1(ind)  #

            if not ind.fitness.valid:
                ind.fitness.values = self.evalfunc(ind)
            single_ind.fitness.values = (ind.fitness.values[target_index],)
            single_population.append(single_ind)
        return single_population

    def convert_to_multi(self, population_single, population_multi):
        """
        This function adds a second evaluation indicator to the single target individual that has undergone the genetic process
        """
        new_population_multi = []

        for single_ind, multi_ind in zip(population_single, population_multi):
            if single_ind.fitness.valid:
                new_multi_ind = creator.Individual(single_ind)

                fitness_values = self.evalfunc(new_multi_ind)
                new_multi_ind.fitness.values = fitness_values

                new_population_multi.append(new_multi_ind)  #
            else:
                raise ValueError("Single target individual fitness is invalid.")
        return new_population_multi

    def single_target_optimization(self, population_multi, toolbox_single, target_index, ngen=1):
        """
        Do single objective optimization,
        based on selectBest algorithm, to rank the offspring list of the returned objective in descending order
        """
        single_population = self.convert_to_single(population_multi, target_index)

        invalid_ind = [ind for ind in single_population if not ind.fitness.valid]
        fitnesses = map(toolbox_single.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        offspring = list(map(toolbox_single.clone, single_population))
        algorithms.eaSimple(offspring, toolbox_single, cxpb=0.8, mutpb=0.2, ngen=ngen, verbose=False)  # 从大到小排列整齐的后代
        single_population[:] = toolbox_single.select(offspring, len(single_population))

        return self.convert_to_multi(single_population, population_multi)

    def run(self):
        """this is the main for genetic process"""
        random.seed(self.random_seed)

        total_population = self.toolbox_multi.population(n=self._population_num)
        hof = tools.HallOfFame(1)

        stats_fit1 = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats_fit1.register("avg", np.mean)
        stats_fit1.register("std", np.std)
        stats_fit1.register("min", np.min)
        stats_fit1.register("max", np.max)

        stats_fit2 = tools.Statistics(lambda ind: ind.fitness.values[1])
        stats_fit2.register("avg", np.mean)
        stats_fit2.register("std", np.std)
        stats_fit2.register("min", np.min)
        stats_fit2.register("max", np.max)

        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(size=stats_size, first_obj=stats_fit1, second_obj=stats_fit2)

        logbook = tools.Logbook()
        logbook.header = mstats.fields

        combined_population = []

        # Batch size, which refers to the number of formulas within a small population
        num_batches = len(total_population) // self.batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = (batch_idx + 1) * self.batch_size

            # Retrieve the population of the current batch
            population = total_population[start_idx:end_idx]

            # run the genetic process
            for gen in range(self.generation):
                now = datetime.now()
                # Optimize target 1 and target 2 separately for the parent generation
                population1 = self.single_target_optimization(population, self.toolbox_single1, target_index=0, ngen=1)
                population2 = self.single_target_optimization(population, self.toolbox_single2, target_index=1, ngen=1)

                # Combine parents and offsprings，number will be 3*self.batch_size
                population = population1 + population2 + population

                # update Hall of Fame
                hof.update(population)

                # record statistic information
                record = mstats.compile(population)
                logbook.record(gen=gen, nevals=len(population), **record)
                print(logbook.stream)

                population = self.toolbox_multi.select(population, self.batch_size)
                after = datetime.now()
                time_dif = after - now
                print(f"runtime: {time_dif.seconds // 3600}h {(time_dif.seconds % 3600) // 60}m {time_dif.seconds % 60}s")

            # SortNondominated return a three-dimensional list of n Pareto frontiers，
            # [[[individual], [individual], []], [[individual], []], ..., [[], [], []]]
            best_individual = tools.sortNondominated(population, len(population), first_front_only=True)[0][0]
            best_fitness = best_individual.fitness.values
            print(f"Batch {batch_idx+1}: Best individual is {best_individual}")
            print(f"Batch {batch_idx+1}: Best individual fitness is {best_fitness}")

            # Extract the first and second Pareto front individuals and combined them into combined population
            population_in_one_two = tools.sortNondominated(population, len(population), first_front_only=False)[0:3]
            try:
                combined_population += population_in_one_two[0]
                combined_population += population_in_one_two[1]
            except:
                combined_population += population_in_one_two[0]

        # Use sortNondominated sorting to find all Pareto frontiers and select the top half of frontiers
        combined_population = {str(ind): ind for ind in combined_population}.values()
        front = tools.sortNondominated(combined_population, len(combined_population), first_front_only=False)

        selected_individuals = []
        if len(front) > 3:
            selected_individuals = [ind for front_individuals in front[:len(front) // 2] for ind in front_individuals]
        else:
            for front_individuals in front[0:2]:
                for ind in front_individuals:
                    selected_individuals.append(ind)

        # Remove duplicate factors by using a dictionary
        unique_individuals = {str(ind): ind for ind in selected_individuals}.values()

        # valid process: further selected by valid set
        final_ind = []
        for ind in list(unique_individuals):
            valid_evaluation = self.evalfunc_valid(ind)
            if valid_evaluation > ind.fitness.values[0] / 2 and valid_evaluation > 0.002 and ind.fitness.values[0] > 0.003:
                print('Accepted:     ', 'train_first_evaluation is {:.5f}'.format(ind.fitness.values[0]), ' / valid_first_evaluation is {:.5f} '.format(float(valid_evaluation)), ind)
                final_ind.append(ind)
            else:
                print('Not Accepted: ', 'train_first_evaluation is {:.5f}'.format(ind.fitness.values[0]), ' / valid_first_evaluation is {:.5f} '.format(float(valid_evaluation)), ind)
        return final_ind
        # new_factor = self.toolbox_multi.compile(expr=ind)
        # print(new_factor)
        # print(new_factor.shape)

    def generate_factor(self, individual):
        """
        Load in the formula, predict,
        and return the predicted factor value in the following dataframe format:：date, token, factor
        """
        factor = self.toolbox_multi.compile_test(expr=individual)
        column, indexes = self.back_to_dataframe(self.test_data)  # Only the data to be predicted should be entered here
        dff = pd.DataFrame(factor)
        dff.columns = column
        dff.index = indexes
        dff = dff.stack().sort_index(level=['token', 'date']).rename(str(individual))
        return dff





