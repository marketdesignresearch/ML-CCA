"""
FILE DESCRIPTION:

This file implements the class WDP (Winner Determination Problem) for GENERIC DOMAINS. 
This class is used for solving a WDP given a finite sample of submitted generic bids.
WDP has the following functionalities:
    0.CONSTRUCTOR: __init__(self, bids)
       bids = list of numpy nxd arrays representing elicited bundle-value pairs from each bidder. n=number of elicited bids, d = number of items + 1(value for that bundle).
       MIP_parameters = dictionary of MIP parameters,
       capacity_generic_items = np.array  of size M, where M is the number of generic items. capacity_generic_items[m] = capacity of generic item m.
    1.METHOD: initialize_mip(self, verbose=False)
        verbose = boolean, level of verbosity when initializing the MIP for the logger.
        This method initializes the winner determination problem as a MIP.
    2.METHOD: solve_mip(self)
        This method solves the MIP of the winner determination problem and sets the optimal allocation.
    3.METHOD: log_solve_details
        This method logs Solution details.
    4.METHOD: __repr__(self)
        Echoe on on your python shell when it evaluates an instances of this class.
    5.METHOD: print_optimal_allocation(self)
        This method printes the optimal allocation x_star in a nice way.
"""

# Libs
import numpy as np
import pandas as pd
import logging
# CPLEX: Here, DOcplex is used for solving the deep neural network-based Winner Determination Problem.
import docplex.mp.model as cpx
# documentation
# http://ibmdecisionoptimization.github.io/docplex-doc/mp/docplex.mp.model.html
# %%


class MLCA_DQ_WDP_GENERIC:

    def __init__(self,
                bids,
                MIP_parameters,
                capacity_generic_items):

        self.bids = bids  # list of numpy nxd arrays representing elicited bundle-value pairs from each bidder. n=number of elicited bids, d = number of items + 1(value for that bundle).
        self.N = len(bids)  # number of bidders
        self.M = bids[0].shape[1] - 1  # number of items
        self.Mip = cpx.Model(name="GENERIC-WDP")  # cplex model
        self.K = [x.shape[0] for x in bids]  # number of elicited bids per bidder
        self.z = {}  # binary decision variables. z(i,k) = 1 <=> bidder i gets the kth bundle out of 1,...,K[i] from his set of bundle-value pairs
        self.x_star = np.zeros((self.N, self.M), dtype=int)  # optimal allocation of the winner determination problem, initialized with zeros
        self.MIP_parameters = MIP_parameters  # MIP parameters

        # NEW
        #--------------------
        self.capacity_generic_items = capacity_generic_items
        #--------------------

    def initialize_mip(self, verbose=0):
        # set MIP parameters
        self.Mip.set_time_limit(self.MIP_parameters['timeLimit'])
        self.Mip.parameters.mip.tolerances.mipgap.set(self.MIP_parameters['MIPGap'])
        self.Mip.parameters.mip.tolerances.integrality.set(self.MIP_parameters['IntFeasTol'])
        self.Mip.parameters.simplex.tolerances.feasibility.set(self.MIP_parameters['FeasibilityTol'])


        for i in range(0, self.N):  # over bidders i \in N
            # add decision variables
            self.z.update({(i, k): self.Mip.binary_var(name="z({},{})".format(i, k)) for k in range(0, self.K[i])})  # z(i,k) = 1 <=> bidder i gets bundle k \in K[i]
            # add allocation constraints for z(i,k)
            self.Mip.add_constraint(ct=(self.Mip.sum(self.z[(i, k)] for k in range(self.K[i])) <= 1), ctname="CT Allocation Bidder {}".format(i))

        # add intersection constraints of bundles for z(i,k)
        for m in range(0, self.M):  # over items m \in M
            self.Mip.add_constraint(ct=(self.Mip.sum(self.z[(i, k)]*self.bids[i][k, m] for i in range(0, self.N) for k in range(0, self.K[i])) <= self.capacity_generic_items[m]), ctname="CT Intersection Item {}".format(m))

        # add objective
        objective = self.Mip.sum(self.z[(i, k)]*self.bids[i][k, self.M] for i in range(0, self.N) for k in range(0, self.K[i]))
        self.Mip.maximize(objective)

        if verbose==1:
            for m in range(0, self.Mip.number_of_constraints):
                    logging.debug('({}) %s'.format(m), self.Mip.get_constraint_by_index(m))
            logging.debug('\nMip initialized')

    def solve_mip(self, verbose=0):
        solution = self.Mip.solve()
        self.solution = solution
        if verbose==1:
            self.log_solve_details(self.Mip)
        # set the optimal allocation
        for i in range(0, self.N):
            for k in range(0, self.K[i]):
                #if self.z[(i, k)].solution_value != 0:
                    #self.x_star[i, :] = self.z[(i, k)].solution_value*self.bids[i][k, :-1]
                if self.z[(i, k)].solution_value > 0.99:
                    self.x_star[i, :] = self.bids[i][k, :-1]

    def log_solve_details(self, solved_mip):
        details = solved_mip.get_solve_details()
        logging.info('Status  : %s', details.status)
        logging.info('Time    : %s sec',round(details.time))
        logging.info('Problem : %s',details.problem_type)
        logging.info('Rel. Gap: {} %'.format(details.mip_relative_gap))
        logging.info('N. Iter : %s',details.nb_iterations)
        logging.info('Hit Lim.: %s',details.has_hit_limit())
        logging.info('Objective Value: %s', solved_mip.objective_value)

    def get_solve_details(self):
        details = self.Mip.get_solve_details()
        unsatisfied_constraints = len(self.solution.find_unsatisfied_constraints(self.Mip))

        details_dict = {'Status': details.status, 'Time': details.time, 'Relative Gap': details.mip_relative_gap, 'Unsatisfied Constraints': unsatisfied_constraints}
        return details_dict

    def summary(self):
        print('################################ OBJECTIVE ################################')
        try:
            print('Objective Value: ', self.Mip.objective_value, '\n')
        except Exception:
            print("Not yet solved!\n")
        print('############################# SOLVE STATUS ################################')
        print(self.Mip.get_solve_details())
        print(self.Mip.get_statistics(), '\n')
        try:
            print(self.Mip.get_solve_status(), '\n')
        except AttributeError:
            print("Not yet solved!\n")
        print('########################### ALLOCATED BIDDERs ############################')
        try:
            for i in range(0, self.N):
                for k in range(0, self.K[i]):
                    if self.z[(i, k)].solution_value != 0:
                        print('z({},{})='.format(i, k), int(self.z[(i, k)].solution_value))
        except Exception:
            print("Not yet solved!\n")
        print('########################### OPT ALLOCATION ###############################')
        self.print_optimal_allocation()
        return(' ')

    def print_optimal_allocation(self):
        D = pd.DataFrame(np.concatenate((self.x_star,self.capacity_generic_items.reshape(1,self.M)),axis=0), index=[f'Bidder_{i}' for i in range(self.N)] +['Capacity'])
        D.columns = ['Item_{}'.format(j) for j in range(self.M)]
        print(D)
