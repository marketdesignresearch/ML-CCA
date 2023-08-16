# Libs
import time
from timeit import default_timer as timer
import gurobipy as gp
from gurobipy import GRB
import numpy as np

#%% NEW SUCCINCT MVNN MIP FOR CALCULATING MAX UTILITY BUNDLE FOR SINGLE BIDDER: argmax_x {MVNN_i(x)-p*x}
class GUROBI_MIP_MVNN_GENERIC_SINGLE_BIDDER_UTIL_MAX:


    def __init__(self,
                 model,
                 #SATS_domain,
                 #bidder_id,
                 #GSVM_national_bidder_goods_of_interest
                 ):
        
        # MVNN PARAMETERS
        self.model = model  # MVNN TORCH MODEL
        self.ts = [layer.ts.data.cpu().detach().numpy().reshape(-1, 1) for layer in model.layers if layer._get_name() not in ['Linear']]
        # MIP VARIABLES
        self.y_variables = []  # POSITIVE INTEGER VARS y[0] and CONT VARS y[i] for i > 0
        self.a_variables = []  # BINARY VARS 1
        self.b_variables = []  # BINARY VARS 2
        self.diff_variables = []  # HELPER CONT VARS (for absolute value in "already-queried-bundle"-constraints)
        self.abs_variables = []  # HELPER POSITVE CONT VARS (for absolute value in "already-queried-bundle"-constraints)
        self.case_counter = {'Case1': 0, 'Case2': 0, 'Case3': 0, 'Case4': 0, 'Case5': 0}
 

    def calc_preactivated_box_bounds(self,
                                     verbose = False):

        # BOX-bounds for y variable (preactivated!!!!) as column vectors

        # Initialize
        input_upper_bounds = np.array(self.model.capacity_generic_goods, dtype=np.int64).reshape(-1, 1)
        self.upper_box_bounds = [input_upper_bounds]
        self.lower_box_bounds = [np.array([0] * self.model.layers[0].in_features, dtype=np.int64).reshape(-1, 1)]

        
        # Propagate through Network 
        for i, layer in enumerate(self.model.layers):

            # -------------------
            if i == 0:
                W = layer.weight.data.cpu().detach().numpy()
                # Remark: no bias b
                # Remark: no vector t since we calculate pre-activate bounds for first hidden layer (generic trafo)
                self.upper_box_bounds.append(W @ self.upper_box_bounds[-1])
                self.lower_box_bounds.append(W @ self.lower_box_bounds[-1])

            elif i == 1:
                W = layer.weight.data.cpu().detach().numpy()
                b = layer.bias.data.cpu().detach().numpy().reshape(-1, 1)
                # Remark: no vector t needed since we calculate pre-activate bounds for second hidden layer (first hidden MVNN layer), thus no self.phi() here
                self.upper_box_bounds.append(W @ self.upper_box_bounds[-1] + b)
                self.lower_box_bounds.append(W @ self.lower_box_bounds[-1] + b)

            else:
                W = layer.weight.data.cpu().detach().numpy()
                b = layer.bias.data.cpu().detach().numpy().reshape(-1, 1)
                # Remark: now we need for preactivated bounds of the ith hidden layer the ts from the previous layer i.e. i-1
                # However, since self.ts has one dimension less than self.model.layers we need to access now (i-2)nd position which has the vector of t for the (i-1)st hidden layer
                t = self.ts[i-2]
                self.upper_box_bounds.append(W @ self.phi(self.upper_box_bounds[-1], t) + b)
                self.lower_box_bounds.append(W @ self.phi(self.lower_box_bounds[-1], t) + b)
            # -------------------

        if verbose:
            print('Upper Box Bounds:')
            print(self.upper_box_bounds)
        if verbose:
            print('Lower Box Bounds:')
            print(self.lower_box_bounds)
        return


    def phi(self, x, t):
        # Bounded ReLU (bReLU) activation function for MVNNS with cutoff t
        return np.minimum(t, np.maximum(0, x)).reshape(-1, 1)


    def generate_mip(self,
                     prices = None,
                     MIPGap = None,
                     verbose = False,
                     ):

        self.mip = gp.Model("MVNN GENERIC MIP")

        # Add IntFeasTol, primal feasibility
        if MIPGap:
            self.mip.Params.MIPGap = MIPGap

        self.calc_preactivated_box_bounds(verbose=verbose)

        # --- Variable declaration -----
        input_ubs = {i: self.upper_box_bounds[0][i, 0] for i in range(len(prices))}
        input_lbs = {i: self.lower_box_bounds[0][i, 0] for i in range(len(prices))}
        self.y_variables.append(self.mip.addVars([i for i in range(len(prices))], name="x_", vtype = GRB.INTEGER, lb=input_lbs, ub=input_ubs))  # the "input variables, i.e. the first y level"

        for (i, layer) in enumerate(self.model.layers):
            tmp_y_variables = []
            tmp_a_variables = []
            tmp_b_variables = []
            # ----------------------------
            for j in range(len(layer.weight.data)):
                if i == 0:
                    # NEW: first hidden layer after generic transformation has no cutoff t and an upper bound of 1
                    # ----------------
                    tmp_y_variables.append(self.mip.addVar(name=f'y_{i+1}_{j}', vtype = GRB.CONTINUOUS, lb = 0, ub = 1))
                    # ----------------
                    # Remark no binary variables for first hidden layer after generic transformation
                else:
                    tmp_y_variables.append(self.mip.addVar(name=f'y_{i+1}_{j}', vtype = GRB.CONTINUOUS, lb = 0, ub = self.ts[i-1][j, 0]))
                    tmp_a_variables.append(self.mip.addVar(name=f'a_{i+1}_{j}', vtype = GRB.BINARY))
                    tmp_b_variables.append(self.mip.addVar( name=f'b_{i+1}_{j}', vtype = GRB.BINARY))
            # ----------------------------
            self.y_variables.append(tmp_y_variables)
            if len(tmp_a_variables) > 0:
                self.a_variables.append(tmp_a_variables)
            if len(tmp_b_variables) > 0:
                self.b_variables.append(tmp_b_variables)

        layer = self.model.output_layer
        self.y_variables.append([self.mip.addVar(name='y_output_0', vtype = GRB.CONTINUOUS, lb = 0)])

        # ---  MVNN Contraints ---
        # Remark: now we need to acces for self.y_variables[i+1] the self.ts[i-1], self.a_variables[i-1] and self.b_variables[i-1] !!!
        for (i, layer) in enumerate(self.model.layers):
            if i == 0:
                # NEW: first hidden layer after generic transformation
                for (j, weight) in enumerate(layer.weight.data):
                    self.y_variables[i+1][j] = gp.quicksum(weight[k] * self.y_variables[i][k] for k in range(len(weight))) # no bias in generic transformation
            else:
                for (j, weight) in enumerate(layer.weight.data):
                    # CASE 1 -> REMOVAL:
                    if self.lower_box_bounds[i+1][j, 0] >= self.ts[i-1][j, 0]:
                        self.y_variables[i+1][j] = self.ts[i-1][j, 0]
                        self.case_counter['Case1'] += 1
                    # CASE 2 -> REMOVAL:
                    elif self.upper_box_bounds[i+1][j, 0] <= 0:
                        self.y_variables[i+1][j] = 0
                        self.case_counter['Case2'] += 1
                    # CASE 3 -> REMOVAL:
                    elif (self.lower_box_bounds[i+1][j, 0] >= 0 and self.lower_box_bounds[i+1][j, 0] <= self.ts[i-1][j, 0]) and (self.upper_box_bounds[i+1][j, 0] >= 0 and self.upper_box_bounds[i+1][j, 0] <= self.ts[i-1][j, 0]):
                        self.y_variables[i+1][j] = gp.quicksum(weight[k] * self.y_variables[i][k] for k in range(len(weight))) + layer.bias.data[j]
                        self.case_counter['Case3'] += 1
                    # CASE 4 -> REMOVAL:
                    elif self.lower_box_bounds[i+1][j, 0] >= 0:
                        # TYPE 1 Constraints for the whole network (except the output layer)
                        self.mip.addConstr(self.y_variables[i+1][j] <= self.ts[i-1][j, 0], name=f'HLayer_{i+1}_{j}_Case4_CT1')
                        # TYPE 2 Constraints for the whole network (except the output layer)
                        self.mip.addConstr(self.y_variables[i+1][j] <= gp.quicksum(weight[k] * self.y_variables[i][k] for k in range(len(weight))) + layer.bias.data[j], name=f'HLayer_{i+1}_{j}_Case4_CT2')
                        # TYPE 3 Constraints for the whole network (except the output layer)
                        self.mip.addConstr(self.y_variables[i+1][j] >= self.b_variables[i-1][j] * self.ts[i-1][j, 0], name=f'HLayer_{i+1}_{j}_Case4_CT3')
                        # TYPE 4 Constraints for the whole network (except the output layer)
                        self.mip.addConstr(self.y_variables[i+1][j] >= gp.quicksum(weight[k] * self.y_variables[i][k] for k in range(len(weight))) + layer.bias.data[j] + (self.ts[i-1][j, 0] - self.upper_box_bounds[i+1][j, 0]) * self.b_variables[i-1][j], name=f'HLayer_{i+1}_{j}_Case4_CT4')
                        self.case_counter['Case4'] += 1
                    # CASE 5 -> REMOVAL:
                    elif self.upper_box_bounds[i+1][j, 0] <= self.ts[i-1][j, 0]:
                        # TYPE 1 Constraints for the whole network (except the output layer)
                        self.mip.addConstr(self.y_variables[i+1][j] <= self.a_variables[i-1][j] * self.ts[i-1][j, 0], name=f'HLayer_{i+1}_{j}_Case5_CT1')
                        # TYPE 2 Constraints for the whole network (except the output layer)
                        self.mip.addConstr(self.y_variables[i+1][j] <= gp.quicksum(weight[k] * self.y_variables[i][k] for k in range(len(weight))) + layer.bias.data[j] - self.lower_box_bounds[i+1][j, 0]*(1-self.a_variables[i-1][j]), name=f'HLayer_{i+1}_{j}_Case5_CT2')
                        # TYPE 3 Constraints for the whole network (except the output layer)
                        self.mip.addConstr(self.y_variables[i+1][j] >= 0, name=f'HLayer_{i+1}_{j}_Case5_CT3')
                        # TYPE 4 Constraints for the whole network (except the output layer)
                        self.mip.addConstr(self.y_variables[i+1][j] >= gp.quicksum(weight[k] * self.y_variables[i][k] for k in range(len(weight))) + layer.bias.data[j], name=f'HLayer_{i+1}_{j}_Case5_CT4')
                        self.case_counter['Case5'] += 1
                    # DEFAULT CASE -> NO REMOVAL:
                    else:
                        # TYPE 1 Constraints for the whole network (except the output layer)
                        self.mip.addConstr(self.y_variables[i+1][j] <= self.a_variables[i-1][j] * self.ts[i-1][j, 0], name=f'HLayer_{i+1}_{j}_Default_CT1')
                        # TYPE 2 Constraints for the whole network (except the output layer)
                        self.mip.addConstr(self.y_variables[i+1][j] <= gp.quicksum(weight[k] * self.y_variables[i][k] for k in range(len(weight))) + layer.bias.data[j] - self.lower_box_bounds[i+1][j, 0]*(1-self.a_variables[i-1][j]), name=f'HLayer_{i+1}_{j}_Default_CT2')
                        # TYPE 3 Constraints for the whole network (except the output layer)
                        self.mip.addConstr(self.y_variables[i+1][j] >= self.b_variables[i-1][j] * self.ts[i-1][j, 0], name=f'HLayer_{i+1}_{j}_Default_CT3')
                        # TYPE 4 Constraints for the whole network (except the output layer)
                        self.mip.addConstr(self.y_variables[i+1][j] >= gp.quicksum(weight[k] * self.y_variables[i][k] for k in range(len(weight))) + layer.bias.data[j] + (self.ts[i-1][j, 0] - self.upper_box_bounds[i+1][j, 0]) * self.b_variables[i-1][j], name=f'HLayer_{i+1}_{j}_Default_CT4')

        output_weight = self.model.output_layer.weight.data[0]
        if (self.model.output_layer.bias is not None):
            output_bias = self.model.output_layer.bias.data
        else:
            output_bias = 0

        if output_bias!=0:
            raise ValueError('output_bias is not 0')

        # Final output layer of MVNN
        # Linear Constraints for the output layer WITH lin_skip_layer: W*y
        if hasattr(self.model, 'lin_skip_layer'):
            lin_skip_W = self.model.lin_skip_layer.weight.detach().cpu().numpy() 
            self.mip.addConstr(gp.quicksum(output_weight[k] * self.y_variables[-2][k] for k in range(len(output_weight))) + output_bias + gp.quicksum(lin_skip_W[0, i]*self.y_variables[0][i] for i in range(lin_skip_W.shape[1])) == self.y_variables[-1][0], name='output_layer')
        # Linear Constraints for the output layer WIHTOUT lin_skip_layer: W*y + W_0*x
        else:
            self.mip.addConstr(gp.quicksum(output_weight[k] * self.y_variables[-2][k] for k in range(len(output_weight))) + output_bias == self.y_variables[-1][0], name='output_layer')

        # --- Objective Declaration ---
        self.mip.setObjective(self.y_variables[-1][0] - gp.quicksum(self.y_variables[0][i] * prices[i] for i in range(len(prices))), GRB.MAXIMIZE)

        self.mip.update()
        if (verbose):
            self.mip.write('MVNN_generic_mip_'+'_'.join(time.ctime().replace(':', '-').split(' '))+'.lp') # remark: if not updated mip.write() also calls mip.update()

        return


    def update_prices_in_objective(self, prices):
        self.mip.setObjective(self.y_variables[-1][0] - gp.quicksum(self.y_variables[0][i] * prices[i] for i in range(len(prices))), GRB.MAXIMIZE)
        return


    def add_forbidden_bundle(self, bundle):

        # NEW: add constraint that the bundle is not already queried for generic bundles TODO: check if this is correct
        # First define helper variables for absolute value
        curr_len = len(self.diff_variables)
        self.diff_variables.append(self.mip.addVars([i for i in range(len(bundle))], name=f"forbidden_bundle{curr_len+1}_abs_helper_var_", vtype = GRB.INTEGER, lb = -GRB.INFINITY)) # models the difference y[0][i] - bundle[i]
        self.abs_variables.append(self.mip.addVars([i for i in range(len(bundle))], name=f"forbidden_bundle{curr_len+1}_abs_var_", vtype = GRB.INTEGER, lb = 0)) # models the absolute value of the difference y[0][i] - bundle[i]
        
        for m in range(len(bundle)):
            # Remark: access with [-1] always the last element of self.diff_variables and self.abs_variables
            self.mip.addConstr(self.diff_variables[-1][m] == (self.y_variables[0][m]-bundle[m]) , name=f'forbidden_bundle{curr_len+1}_diff_CT_{m}')
            self.mip.addConstr(self.abs_variables[-1][m] == gp.abs_(self.diff_variables[-1][m]), name=f'forbidden_bundle{curr_len+1}_abs_CT_{m}')
        self.mip.addConstr(gp.quicksum(self.abs_variables[-1][m] for m in range(len(bundle))) >= 1, name=f'alreadyQueried_bundle{curr_len+1}')
        # -----------------------------

        self.mip.update()
        return


    def solve_mip(self,
                  outputFlag = False,
                  verbose = True,
                  timeLimit = np.inf,
                  MIPGap = 1e-04,
                  IntFeasTol = 1e-5,
                  FeasibilityTol = 1e-6
                  ):
        
        if not verbose:
            self.mip.Params.LogToConsole = 0
            self.mip.Params.OutputFlag = 0

        # set solve parameter (if not sepcified, default values are used)
        self.mip.Params.timeLimit = timeLimit # Default +inf
        self.mip.Params.MIPGap = MIPGap # Default 1e-04
        self.mip.Params.IntFeasTol = IntFeasTol # Default 1e-5
        self.mip.Params.FeasibilityTol = FeasibilityTol # Default 1e-6
        #

        self.start = timer()
        self.mip.Params.OutputFlag = outputFlag
        self.mip.optimize() # remark: if not updated mip.optimize() also calls mip.update()
        self.end = timer()

        self.optimal_schedule = []
        # test try-catch for non-feasible solution
        try:
            for i in range(len(self.y_variables[0])):
                self.optimal_schedule.append(np.int64(np.round(self.y_variables[0][i].x, decimals=0))) # TODO: check if this is correct
        except:
            self._print_info()
            raise ValueError('MIP did not solve succesfully!')

        if verbose:
            self._print_info()

        return self.optimal_schedule


    def _print_info(self):
        print(*['*']*30)
        print('MIP INFO:')
        print(*['-']*30)
        print(f'Name: {self.mip.ModelName}')
        print(f'Goal: {self._model_sense_converter(self.mip.ModelSense)}')
        print(f'Objective: {self.mip.getObjective()}')
        print(f'Number of variables: {self.mip.NumVars}')
        print(f' - Binary {self.mip.NumBinVars}')
        print(f'Number of linear constraints: {self.mip.NumConstrs}')
        print(f'Primal feasibility tolerance for constraints: {self.mip.Params.FeasibilityTol}')
        print(f'Integer feasibility tolerance: {self.mip.Params.IntFeasTol}')
        print(f'Relative optimality gap: {self.mip.Params.MIPGap}')  # we may want this 
        print(f'Time Limit: {self.mip.Params.TimeLimit}')
        print('')
        print('MIP SOLUTION:')
        print(*['-']*30)
        print(f'Status: {self._status_converter(self.mip.status)}')
        print(f'Elapsed in sec: {self.end - self.start}')
        print(f'Reached Relative optimality gap: {self.mip.MIPGap}')   
        print(f'Optimal Allocation: {self.optimal_schedule}')
        print(f'Objective Value: {self.mip.ObjVal}')
        print(f'Number of stored solutions: {self.mip.SolCount}')
        print('IA Case Statistics:')
        for k, v in self.case_counter.items():
            print(f' - {k}: {v}')
        print(*['*']*30)


    def _status_converter(self, int_status):
        status_table = ['woopsies!', 'LOADED', 'OPTIMAL', 'INFEASIBLE', 'INF_OR_UNBD', 'UNBOUNDED', 'CUTOFF', 'ITERATION_LIMIT', 'NODE_LIMIT', 'TIME_LIMIT', 'SOLUTION_LIMIT', 'INTERRUPTED', 'NUMERIC', 'SUBOPTIMAL', 'INPROGRESS', 'USER_OBJ_LIMIT']
        return status_table[int_status]


    def _model_sense_converter(self, int_sense):
        if int_sense == 1:
            return 'Minimize'
        elif int_sense == -1:
            return 'Maximize'
        else:
            raise ValueError('int_sense needs to be -1:maximize or 1: minimize')