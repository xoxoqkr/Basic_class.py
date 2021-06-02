# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random
import ASP_code.ASP_class as ASP_class
#from docplex.mp.model import Model

def SimpleAssingmentSolver(driver_set, customers_set, v_old, print_gurobi=False):
    drivers = range(len(driver_set))
    customers = range(len(customers_set))
    # v_old  계산기 추가
    # Create a new model
    org_m = gp.Model("mip1")
    # Create variables
    x = org_m.addVars(len(drivers), len(customers), vtype=GRB.BINARY, name="x")
    # obj
    org_m.setObjective(gp.quicksum(v_old[i, j] * x[i, j] for i in drivers for j in customers), GRB.MAXIMIZE)
    # Add constraint: 차량은 1개의 고객만 서비스 가능
    for i in drivers:
        org_m.addConstr(gp.quicksum(x[i, j] for j in customers) == 1)
    # Add constraint: 고객은 1개의 차량에 의해서만 서비스 받을 수 있음.
    for j in customers:
        org_m.addConstr(gp.quicksum(x[i, j] for i in drivers) <= 1)
    # Optimize model
    res = []
    if print_gurobi == False:
        org_m.setParam(GRB.Param.OutputFlag, 0)
    print("VARS", org_m.getVars())
    org_m.optimize()
    for v in org_m.getVars():
        res.append(abs(v.x))
    res2 = np.array(res)
    res2 = res2.reshape(len(drivers), len(customers))
    return res2, org_m.objVal


def SimpleInverseSolverwithX(driver_set, customers_set, x, v_old, old_obj, lower_b=-0.2, upper_b=False, print_gurobi=False):
    drivers = list(range(len(driver_set)))
    customers = list(range(len(customers_set)))
    m = gp.Model("mip1")
    v = m.addVars(len(drivers), len(customers), vtype=GRB.CONTINUOUS, name="v")  # GRB.INTEGER
    z = m.addVars(len(drivers), len(customers), vtype=GRB.CONTINUOUS, name="z")  # GRB.INTEGER
    m.setObjective(gp.quicksum(z[i, j] for i in drivers for j in customers), GRB.MINIMIZE)
    m.addConstrs(z[i, j] - v_old[i,j] == gp.abs_(v[i, j]) for i in drivers for j in customers) #z = abs(v)
    m.addConstr(gp.quicksum(v[i,j]*x[i,j] for i in drivers for j in customers) <= old_obj)
    m.addConstrs( v[i, j] >= lower_b*v_old[i,j] for i in drivers for j in customers)
    if upper_b != False:
        m.addConstrs(v[i, j] <= upper_b for i in drivers for j in customers)
    if print_gurobi == False:
        m.setParam(GRB.Param.OutputFlag, 0)
    m.optimize()
    res = printer(m.getVars(), [], len(drivers), len(customers))
    return res



def SimpleInverseSolver(driver_set, customers_set, x_old, v_old, lower_b=False, upper_b=False, sp=None,
                        data_record=False, print_gurobi=False, delta=500):
    drivers = list(range(len(driver_set)))
    customers = list(range(len(customers_set)))
    # drivers_name = NamesReturner(driver_set)
    # customers_name = NamesReturner(customers_set)
    m = gp.Model("mip1")
    x = m.addVars(len(drivers), len(customers), vtype=GRB.BINARY, name="x")
    v = m.addVars(len(drivers), len(customers), vtype=GRB.CONTINUOUS, name="v")  # GRB.INTEGER
    # b = m.addVars(len(drivers),len(customers), vtype=GRB.CONTINUOUS, name = "v")
    rev_v = m.addVars(len(drivers), len(customers), vtype=GRB.CONTINUOUS, name="rev_v")  # GRB.INTEGER
    z = m.addVars(len(drivers), vtype=GRB.CONTINUOUS, name="z")
    obj_abs_val = m.addVars(len(drivers), len(customers), vtype=GRB.CONTINUOUS, name="abs")  # GRB.INTEGER
    print("z", len(z), z[0])
    if sp == None:
        num_sp = max(int(len(drivers) / 2), random.choice(drivers))
        sp = random.sample(customers, num_sp)
        rev_sp = sp
    # req_sp_num = random.choice(range(max(1,len(sp))))
    req_sp_num = min(len(driver_set), len(sp))
    print("sp", rev_sp)
    # sp를 재index
    # rev_sp = ReIndexer(sp, customers_name)
    print("SimpleInverseSolver info")
    print("driver#:", len(drivers), "//customers#:", len(customers), "sp:", len(sp), sp[:3], "...", sp[-3:])
    # Set objective
    # m.setObjective(gp.quicksum(v), GRB.MINIMIZE)
    m.setObjective(gp.quicksum(obj_abs_val[i, j] * x[i, j] for i in drivers for j in customers), GRB.MINIMIZE)

    m.addConstrs(obj_abs_val[i, j] == gp.abs_(v[i, j]) for i in drivers for j in customers)

    m.addConstr(gp.quicksum(x[i, j] for i in drivers for j in rev_sp) >= req_sp_num)
    # Add constraint: 차량은 1명의 고객만 할당 가능.
    for i in drivers:
        m.addConstr(gp.quicksum(x[i, j] for j in customers) == 1)
    # Add constraint: 고객은 1개의 차량에 의해서만 서비스 받을 수 있음.
    for j in customers:
        m.addConstr(gp.quicksum(x[i, j] for i in drivers) <= 1)
    # Add constraint: 선택되는 v 가 v_old 보다는 커야함. -> optimal에 방해 요소.
    for i in drivers:
        for j in customers:
            m.addConstr(rev_v[i, j] == v[i, j] + v_old[i, j])
        # m.addConstr(gp.quicksum((v[i,j] + v_old[i,j])*x[i,j] for j in customers) == gp.max_(rev_v[i,j] for j in customers))
        m.addConstr(z[i] == gp.max_(rev_v[i, j] for j in customers))
        m.addConstr(gp.quicksum((v[i, j] + v_old[i, j]) * x[i, j] for j in customers) == z[i])
    # Add constraint: lower <= rev_v <= upeer value
    for i in drivers:
        for j in customers:
            if lower_b != False:
                m.addConstr(lower_b <= v[i, j])
            if upper_b != False:
                m.addConstr(v[i, j] <= upper_b)
                # Optimize model
    if print_gurobi == False:
        m.setParam(GRB.Param.OutputFlag, 0)

    m.optimize()

    res = []
    res_x = []
    res_v = []
    res_rev_v = []
    index = 0
    # print(m.getVars())
    for v in m.getVars():
        res.append(v.x)
        # print(v, v.VarName)
        if index < len(drivers) * len(customers):
            # if v.VarName == "x":
            res_x.append(abs(v.x))
        elif len(drivers) * len(customers) <= index < 2 * len(drivers) * len(customers):
            res_v.append(v.x)
        elif 2 * len(drivers) * len(customers) <= index < 3 * len(drivers) * len(customers):
            res_rev_v.append(v.x)
        else:
            pass
        index += 1
    res_x = np.array(res_x).reshape(len(drivers), len(customers))
    res_v = np.array(res_v).reshape(len(drivers), len(customers))
    res_rev_v = np.array(res_rev_v).reshape(len(drivers), len(customers))
    record = []
    if data_record == True:
        served_candi = []
        x_old_served = []
        for x in res_x:
            if sum(x) == 1:
                ct_name = list(x).index(1)
                served_candi.append(ct_name)
        record = [sp, served_candi, res_v.sum() - v_old.sum()]
        print("old_sp", served_candi, "rev sum", res_v.sum())
    # tem_v_delta 갱신
    # tem_v_delta = Array2List(res_v, drivers_name, customers_name)
    return [res_x, res_v, res_rev_v, record], m.getVars()  # , tem_v_delta


def printer(data, var_name, num1, num2, data_record=True):
    # num1 은 drivers, num2는 customers
    res = []
    res_x = []
    res_v = []
    c_seq_res = []
    res_y = []
    res_k = []
    res_s = []
    index = 0
    #print('solver check',data[:5])
    for val in data:
        if val.VarName[0] == 'x':
            #print('val',val)
            res_x.append(int(val.x))
        elif val.VarName[0] == 'v':
            res_v.append(int(val.x))
        elif val.VarName[0] == 'c':
            c_seq_res.append(val.x)
        elif val.VarName[0] == 'y':
            res_y.append(val.x)
        elif val.VarName[0] == 'k':
            res_k.append(val.x)
        elif val.VarName[0] == 's':
            res_s.append(val.x)
        else:
            pass
        index += 1
    # res_x = None
    # res_v = None
    # res_y = None
    # res_k = None
    # res_s = None
    if len(res_x) > 1:
        res_x = np.array(res_x).reshape(num1, num2)
    if len(res_v) > 1:
        res_v = np.array(res_v).reshape(num1, num2)
    if len(res_y) > 1:
        res_y = np.array(res_y).reshape(num1, num2)
    if len(res_k) > 1:
        res_k = np.array(res_k).reshape(num1, num2)
    if len(res_s) > 1:
        res_s = np.array(res_s).reshape(num1, num2)
    header = ["res_x", "res_v", "res_y", "res_k", "res_seq_c", "res_s"]
    infos = [res_x, res_v, res_y, res_k, c_seq_res, res_s]
    #for i in range(len(header)):
    #    print(header[i])
    #    print(infos[i])
    return [res_x, res_v]


# 20.07.07. 확인한 모형
def SimpleInverseSolver2(driver_set, customers_set, x_old, v_old, d_orders, lower_b=False, upper_b=False, sp=None,
                         data_record=False, print_gurobi=False, delta=500):
    drivers = list(range(len(driver_set)))
    customers = list(range(len(customers_set)))
    # D.V. and model set.
    m = gp.Model("mip1")
    x = m.addVars(len(drivers), len(customers), vtype=GRB.BINARY, name="x")
    v = m.addVars(len(drivers), len(customers), vtype=GRB.CONTINUOUS, name="v")  # GRB.INTEGER
    rev_v = m.addVars(len(drivers), len(customers), vtype=GRB.CONTINUOUS, name="rev_v")  # GRB.INTEGER
    z = m.addVars(len(drivers), vtype=GRB.CONTINUOUS, name="z")
    obj_abs_val = m.addVars(len(drivers), len(customers), vtype=GRB.CONTINUOUS, name="abs")  # GRB.INTEGER
    c_seq = m.addVars(len(customers), vtype=GRB.INTEGER, name="c_seq")
    y = m.addVars(len(drivers), len(customers), vtype=GRB.BINARY, name="y")
    d = m.addVars(len(drivers), vtype=GRB.CONTINUOUS, name="d")
    if sp == None:
        num_sp = max(int(len(drivers) / 2), random.choice(drivers))
        sp = random.sample(customers, num_sp)
        rev_sp = sp
    req_sp_num = min(len(driver_set), len(sp))
    print("z", len(z), z[0], "sp", rev_sp)
    print("SimpleInverseSolver info", "driver#:", len(drivers), "//customers#:", len(customers), "sp:", len(sp), sp[:3],
          "...", sp[-3:])
    # Set objective
    m.setObjective(gp.quicksum(obj_abs_val[i, j] * x[i, j] for i in drivers for j in customers), GRB.MINIMIZE)
    # const 1. 목적식의 절대값.
    m.addConstrs(obj_abs_val[i, j] == gp.abs_(v[i, j]) for i in drivers for j in customers)

    # const 2. req_sp_num 만큼 m고객을 서비스 해야 함.
    m.addConstr(gp.quicksum(x[i, j] for i in drivers for j in rev_sp) >= req_sp_num)
    # const 3 : 차량은 1명의 고객만 할당 가능.
    m.addConstrs(gp.quicksum(x[i, j] for j in customers) == 1 for i in drivers)
    # const 4: 고객은 1개의 차량에 의해서만 서비스 받을 수 있음.
    m.addConstrs(gp.quicksum(x[i, j] for i in drivers) <= 1 for j in customers)
    # const 5: D.V. rev_v[i,j]는 v[i,j] + v_old[i,j]
    m.addConstrs(rev_v[i, j] == v[i, j] + v_old[i, j] for i in drivers for j in customers)
    # const 6: d_orders[i] 와 c_seq[j], sync 함수
    m.addConstrs((x[i, j] == 1) >> (d_orders[i] == c_seq[j]) for i in drivers for j in customers)
    # const 7: d[i] 계산
    m.addConstrs((d[i] == gp.quicksum(rev_v[i, j] * x[i, j] for j in customers)) for i in drivers)
    # const 8,9: d[i] 계산
    m.addConstrs((y[i, j] == 1) >> (d_orders[i] <= c_seq[j]) for i in drivers for j in customers)
    m.addConstrs((y[i, j] == 1) >> (d[i] <= rev_v[i, j]) for i in drivers for j in customers)
    # m.addConstrs((y[i,j] == 1) >> (d_orders[i] >= c_seq[j]) for i in drivers for j in customers)
    # m.addConstrs((y[i,j] == 1) >> (d[i] >= rev_v[i,j]) for i in drivers for j in customers)
    # Aconstr 7 : Add constraint: lower <= rev_v <= upeer value
    for i in drivers:
        for j in customers:
            if lower_b != False:
                m.addConstr(lower_b <= v[i, j])
            if upper_b != False:
                m.addConstr(v[i, j] <= upper_b)

                # Optimize model
    if print_gurobi == False:
        m.setParam(GRB.Param.OutputFlag, 0)

    m.optimize()
    res = printer(m.getVars(), [], len(drivers), len(customers))
    print('Obj val: %g' % m.objVal, "Solver", solver)
    return res


# 20.07.10 확인 버전 -> 문서화 된 버전
def SimpleInverseSolver3(driver_set, customers_set, v_old, d_orders, ava_match = [], lower_b = False, upper_b = False, sp=None,
                         data_record=False, print_gurobi=False, delta=500, solver=-1, slack = 0, minimum_fee = None):
    drivers = list(range(len(driver_set)))
    customers = list(range(len(customers_set)))
    # D.V. and model set.
    m = gp.Model("mip1")
    x = m.addVars(len(drivers), len(customers), vtype=GRB.BINARY, name="x")
    v = m.addVars(len(drivers), len(customers), vtype=GRB.CONTINUOUS, name="v")  # GRB.INTEGER
    y = m.addVars(len(drivers), len(customers), vtype=GRB.BINARY, name="y")
    c_seq = m.addVars(len(customers), vtype=GRB.INTEGER, name="c")
    d = m.addVars(len(drivers), vtype=GRB.CONTINUOUS, name="d")
    if sp == None:
        num_sp = max(int(len(drivers) / 2), random.choice(drivers))
        sp = random.sample(customers, num_sp)
    req_sp_num = min(len(driver_set), len(sp))
    rev_sp = sp
    print("SimpleInverseSolver info", "driver#:", len(drivers), "//customers#:", len(customers), "sp:", len(sp), sp)
    # Set objective
    m.setObjective(gp.quicksum(v[i, j] for i in drivers for j in customers), GRB.MINIMIZE)

    # constr 1. 사전에 할당된 고객 수행.
    m.addConstr(gp.quicksum(x[i, j] for i in drivers for j in rev_sp) >= req_sp_num)
    # constr 2 : 차량은 1명의 고객만 할당 가능.
    m.addConstrs(gp.quicksum(x[i, j] for j in customers) == 1 for i in drivers)
    # constr 4: 고객은 1개의 차량에 의해서만 서비스 받을 수 있음.
    m.addConstrs(gp.quicksum(x[i, j] for i in drivers) <= 1 for j in customers)
    # 순서 관련 제약식
    m.addConstrs(
        (x[i, j] == 1) >> (d_orders[i] == c_seq[j]) for i in drivers for j in customers)  # <- indicator constraint
    # 변경된 제약식
    m.addConstr(
        gp.quicksum(c_seq[j] for j in customers) == sum(list(range(1, len(drivers) + 1))) + (len(drivers) + 1) * (
                    len(customers_set) - len(drivers)))
    m.addConstrs(c_seq[j] <= len(drivers) + 1 for j in customers)
    m.addConstrs((y[i, j] == 1) >> (d_orders[i] <= c_seq[j]) for i in drivers for j in customers)
    m.addConstrs((y[i, j] == 0) >> (d_orders[i] - 1 >= c_seq[j]) for i in drivers for j in customers)
    m.addConstrs((d[i] == gp.quicksum((v[i, j] + v_old[i, j]) * x[i, j] for j in customers)) for i in drivers)
    m.addConstrs((y[i, j] == 1) >> (d[i] >= v[i, j] + v_old[i, j] + slack) for i in drivers for j in customers)
    # Aconstr 7 : Add constraint: lower <= rev_v <= upeer value
    for i in drivers:
        for j in customers:
            if lower_b != False:
                m.addConstr(lower_b <= v[i, j])
            if upper_b != False:
                m.addConstr(v[i, j] <= upper_b)
                # Optimize model
    if minimum_fee != None:
        m.addConstrs((x[i, j] == 1) >> (v[i, j] >= minimum_fee) for i in drivers for j in sp)
    if len(ava_match) > 0:
        m.addConstrs(x[i,j] <= ava_match[i,j] for i in drivers for j in customers)
    if print_gurobi == False:
        m.setParam(GRB.Param.OutputFlag, 0)
    m.Params.method = solver  # -1은 auto dedection이며, 1~5에 대한 차이.
    m.optimize()
    res = printer(m.getVars(), [], len(drivers), len(customers))
    print(d_orders)
    print('Obj val: %g' % m.objVal, "Solver", solver)
    return res, m.getVars()


# 20.07.25 확인 버전 -> 고객에게 요금을 받는 버전
def SimpleInverseSolver4(driver_set, customers_set, v_old, d_orders, willing_to_pay = None,lower_b=False, upper_b=False, sp=None, print_gurobi=False, solver=-1):
    drivers = list(range(len(driver_set)))
    customers = list(range(len(customers_set)))
    # D.V. and model set.
    m = gp.Model("mip1")
    x = m.addVars(len(drivers), len(customers), vtype=GRB.BINARY, name="x")
    v = m.addVars(len(drivers), len(customers), vtype=GRB.CONTINUOUS, name="v")  # GRB.INTEGER
    y = m.addVars(len(drivers), len(customers), vtype=GRB.BINARY, name="y")
    # obj_abs_val = m.addVars(len(drivers),len(customers), vtype = GRB.CONTINUOUS, name = "abs") # GRB.INTEGER
    c_seq = m.addVars(len(customers), vtype=GRB.INTEGER, name="c")
    d = m.addVars(len(drivers), vtype=GRB.CONTINUOUS, name="d")
    k = m.addVars(len(drivers), len(customers), vtype=GRB.BINARY, name="k")
    s = m.addVars(len(drivers), len(customers), vtype=GRB.BINARY, name="s")

    if sp == None:
        num_sp = max(int(len(drivers) / 2), random.choice(drivers))
        sp = random.sample(customers, num_sp)
        rev_sp = sp
    else:
        rev_sp = sp
    req_sp_num = min(len(driver_set), len(sp))

    p = []
    h = []
    for ite in range(0, len(drivers) * len(customers)):
        if willing_to_pay == None:
            p.append(1)
        else:
            p.append(random.randrange(700, 1000))
        # h.append(random.randrange(500,700))
    p = np.array(p).reshape(len(drivers), len(customers))
    # h = np.array(h).reshape(len(drivers),len(customers))
    for ite in range(0, len(customers)):
        h.append(random.randrange(500, 700))
    # h = np.array(h).reshape(1,len(customers))
    print("h", h)
    Q = 40
    mtr1 = []
    for ite in range(0, len(drivers) * len(customers)):
        val = random.randrange(10, 25)
        mtr1.append(val)
    mtr = np.array(mtr1).reshape(len(drivers), len(customers))
    print("SimpleInverseSolver info", "driver#:", len(drivers), "//customers#:", len(customers), "sp:", len(sp), sp[:3],
          "...", sp[-3:])
    # Set objective
    m.setObjective(gp.quicksum(p[i, j] * x[i, j] + h[j] * s[i, j] - v[i, j] for i in drivers for j in customers),
                   GRB.MAXIMIZE)
    # m.setObjective(gp.quicksum((p[i,j] - v[i,j] + s[i,j]*h[i,j])*x[i,j]  for i in drivers for j in customers), GRB.MAXIMIZE)
    # constr  1: 목적식 sub constraints

    # constr  2: 우선 고객 서비스 제약
    m.addConstr(gp.quicksum(x[i, j] for i in drivers for j in rev_sp) >= req_sp_num)
    # constr 3 : 차량 고객 1대 1 할당 제약
    m.addConstrs(gp.quicksum(x[i, j] for j in customers) == 1 for i in drivers)
    m.addConstrs(gp.quicksum(x[i, j] for i in drivers) <= 1 for j in customers)
    # constr 4 : 서비스 시간 제약식
    m.addConstrs(x[i, j] * mtr[i, j] <= Q for i in drivers for j in customers)
    # m.addConstrs((x[i,j] == 1) >> (mtr[i,j] <= Q ) for i in drivers for j in customers )
    m.addConstrs(s[i, j] * mtr[i, j] <= 0.8 * Q for i in drivers for j in customers)
    # m.addConstrs((s[i,j] == 1) >> (mtr[i,j] <= 0.8*Q) for i in drivers for j in customers )
    m.addConstrs(x[i, j] >= s[i, j] for i in drivers for j in customers)
    # m.addConstrs((x[i,j] == 0) >> (s[i,j] == 0) for i in drivers for j in customers)

    # 순서 관련 제약식
    m.addConstrs(
        (x[i, j] == 1) >> (d_orders[i] == c_seq[j]) for i in drivers for j in customers)  # <- indicator constraint
    m.addConstr(gp.quicksum(c_seq[j] for j in customers) == sum(list(range(1, len(drivers) + 1))))
    m.addConstrs((y[i, j] == 1) >> (d_orders[i] <= c_seq[j]) for i in drivers for j in customers)
    m.addConstrs((y[i, j] == 0) >> (d_orders[i] - 1 >= c_seq[j]) for i in drivers for j in customers)
    m.addConstrs((k[i, j] == 1) >> (y[i, j] + c_seq[j] == 0) for i in drivers for j in customers)
    m.addConstrs((k[i, j] == 0) >> (y[i, j] + c_seq[j] >= 1) for i in drivers for j in customers)
    m.addConstrs((k[i, j] == 1) >> (d[i] >= v[i, j] + v_old[i, j]) for i in drivers for j in customers)
    m.addConstrs((d[i] == gp.quicksum((v[i, j] + v_old[i, j]) * x[i, j] for j in customers)) for i in drivers)
    m.addConstrs((y[i, j] == 1) >> (d[i] >= v[i, j] + v_old[i, j]) for i in drivers for j in customers)

    # m.addConstrs((y[i,j] == 1) >> (c_seq[j]  ==  0 ) for i in drivers for j in customers)
    # Aconstr 7 : Add constraint: lower <= rev_v <= upeer value
    for i in drivers:
        for j in customers:
            if lower_b != False:
                m.addConstr(lower_b <= v[i, j])
            if upper_b != False:
                m.addConstr(v[i, j] <= upper_b)

                # Optimize model
    if print_gurobi == False:
        m.setParam(GRB.Param.OutputFlag, 0)
    m.Params.method = solver  # -1은 auto dedection이며, 1~5에 대한 차이.
    m.optimize()
    res = printer(m.getVars(), [], len(drivers), len(customers))
    print('Obj val: %g' % m.objVal, "Solver", solver)
    return res, m.getVars()  # , tem_v_delta

def InversePhase1(riders, riders_order, customer_set):
    """
    RiderSelectionExpect
    라이더들의 정보를 받아서, 다음 시간에 해당 라이더들이 어떠한 선택을 할지를 계산.
    라이더들이 보조금이 없을 때, 선택하는 고객들의 가치를 계산.
    :param riders: 라이더 class
    :param riders_order: 해당 라이더들의 선택 순서
    :param customer_set: 고객 class
    :result res1 : 입력된 riders_order에 맞는 고객 가치
    """
    res = []
    selected_ct_names = []
    customers = []
    for i in customer_set.keys():
        customers.append(customer_set[i])
    for rider_index in riders_order:
        rider = riders[rider_index]
        #print(customers)
        orders = ASP_class.PriorityOrdering(rider, customers)
        for order in orders:
            if order[0] not in selected_ct_names:
                selected_ct_names.append(orders[0][0])
                res.append([rider.name, int(order[1])]) #todo : value를 프린트
                break
            else:
                pass
    res1 = []
    for info in res:
        res1.append(info[1])
    return res1


def InversePhase2(driver_set, customers_set, v_old, minimum_subsidy, ava_match = [], lower_b = False, upper_b = False, sp=None,
                  print_gurobi=False,  solver=-1, slack = 0, minimum_fee = None):
    """

    :param driver_set: 가능한 드라이버 이름들 list
    :param customers_set: 가능한 고객들 이름들 list
    :param v_old:
    :param minimum_subsidy:
    :param ava_match:
    :param lower_b:
    :param upper_b:
    :param sp:
    :param print_gurobi:
    :param solver:
    :param slack:
    :param minimum_fee:
    :return:
    """
    drivers = list(range(len(driver_set)))
    customers = list(range(len(customers_set)))
    int_upper_b = max(1,upper_b)
    # D.V. and model set.
    m = gp.Model("mip1")
    x = m.addVars(len(drivers), len(customers), vtype=GRB.BINARY, name="x")
    v = m.addVars(len(drivers), len(customers), vtype=GRB.CONTINUOUS, name="v")  # GRB.INTEGER 새로운 z.
    if sp == None:
        num_sp = max(int(len(drivers) / 2), random.choice(drivers))
        sp = random.sample(customers, num_sp)
    req_sp_num = min(len(driver_set), len(sp))
    rev_sp = sp
    print("SimpleInverseSolver info", "driver#:", len(drivers), "//customers#:", len(customers), "sp:", len(sp), sp)
    # Set objective
    m.setObjective(gp.quicksum(v[i, j] for i in drivers for j in customers), GRB.MINIMIZE)
    # constr 1. 사전에 할당된 고객 수행.
    m.addConstr(gp.quicksum(x[i, j] for i in drivers for j in rev_sp) >= req_sp_num)
    # constr 2 : 차량은 1명의 고객만 할당 가능.
    m.addConstrs(gp.quicksum(x[i, j] for j in customers) == 1 for i in drivers)
    # constr 4: 고객은 1개의 차량에 의해서만 서비스 받을 수 있음.
    m.addConstrs(gp.quicksum(x[i, j] for i in drivers) <= 1 for j in customers)
    # 변경된 제약식2
    """
    for i in drivers:
        for j in customers:
            print(i,j,v[i,j],x[i,j],minimum_subsidy[i],v_old[i,j])
            m.addConstr(v[i,j] >= x[i,j]*(minimum_subsidy[i] - v_old[i,j]))
    """
    m.addConstrs(v[i,j] >= x[i,j]*(minimum_subsidy[i] - v_old[i,j]) for i in drivers for j in customers)
    #m.addConstrs(x[i,j] >= v[i,j]/int_upper_b for i in drivers for j in customers)
    #m.addConstrs((d[i] == gp.quicksum((v[i, j] + v_old[i, j]) * x[i, j] for j in customers)) for i in drivers)
    # Aconstr 7 : Add constraint: lower <= rev_v <= upeer value
    for i in drivers:
        for j in customers:
            if lower_b != False:
                m.addConstr(lower_b <= v[i, j])
            if upper_b != False:
                m.addConstr(v[i, j] <= upper_b)
                # Optimize model
    if minimum_fee != None:
        m.addConstrs((x[i, j] == 1) >> (v[i, j] >= minimum_fee) for i in drivers for j in sp)
    if len(ava_match) > 0:
        m.addConstrs(x[i,j] <= ava_match[i,j] for i in drivers for j in customers)
    if print_gurobi == False:
        m.setParam(GRB.Param.OutputFlag, 0)
    m.Params.method = solver  # -1은 auto dedection이며, 1~5에 대한 차이.
    m.optimize()
    res = printer(m.getVars(), [], len(drivers), len(customers))
    print('Obj val: %g' % m.objVal, "Solver", solver)
    return res, m.getVars()

def LinearizedSubsidyProblem(driver_set, customers_set, v_old, ro, lower_b = False, upper_b = False, sp=None, print_gurobi=False,  solver=-1, delta = 100):
    """
    선형화된 버전의 보조금 문제
    :param driver_set: 가능한 라이더 수
    :param customers_set: 가능한 고객 수
    :param v_old: 가치
    :param ro: 라이더 선택 순서
    :param lower_b:
    :param upper_b:
    :param sp:
    :param print_gurobi:
    :param solver:
    :return:
    """
    drivers = list(range(len(driver_set)))
    customers = list(range(len(customers_set)))
    driver_num = len(driver_set)
    customer_num = len(customers_set)
    sum_i = sum(list(range(driver_num)))
    if upper_b == False:
        upper_b = 1000
    if lower_b == False:
        lower_b = 0
    # D.V. and model set.
    m = gp.Model("mip1")
    x = m.addVars(len(drivers), len(customers), vtype=GRB.BINARY, name="x")
    v = m.addVars(len(drivers), len(customers), lb = 0, vtype=GRB.CONTINUOUS, name="v")
    cso = m.addVars(len(customers), vtype=GRB.INTEGER, name="c" )
    #선형화를 위한 변수
    y = m.addVars(len(drivers), len(customers), vtype=GRB.BINARY, name="y")
    w = m.addVars(len(drivers), len(customers), vtype=GRB.CONTINUOUS, name="w")
    z = m.addVars(len(drivers), len(customers), vtype=GRB.CONTINUOUS, name="z")
    b = m.addVars(len(drivers), len(customers), vtype=GRB.CONTINUOUS, name="b") #크시
    #우선 고객 할당.
    if sp == None:
        num_sp = max(int(len(drivers) / 2), random.choice(drivers))
        sp = random.sample(customers, num_sp)
    req_sp_num = min(len(driver_set), len(sp))
    rev_sp = sp
    #print("Priority Customer", rev_sp)
    # Set objective #29
    m.setObjective(gp.quicksum(v[i, j] for i in drivers for j in customers), GRB.MINIMIZE)
    #30
    """
    for i in drivers:
        for j in customers:
            m.addConstr(gp.quicksum(w[i, k] + v_old[i, k] * x[i, k] for k in customers) >= z[i, j] + v_old[i, j] * y[i, j])
    """
    m.addConstrs(gp.quicksum(w[i,k] + v_old[i,k]*x[i,k] for k in customers) >= z[i,j] + v_old[i,j]*y[i,j] + delta for i in drivers for j in customers)
    #31
    m.addConstrs( w[i,j]-v[i,j ]<= upper_b*(1-x[i,j]) for i in drivers for j in customers)
    #32
    m.addConstrs(v[i, j] - w[i, j] <= upper_b*(1 - x[i, j]) for i in drivers for j in customers)
    #33
    m.addConstrs(w[i, j] <= upper_b * x[i, j] for i in drivers for j in customers)
    #34
    m.addConstrs(z[i, j] - v[i, j] <= upper_b*(1 - y[i, j]) for i in drivers for j in customers)
    #35
    m.addConstrs(v[i, j] - z[i, j] <= upper_b * (1 - y[i, j]) for i in drivers for j in customers)
    #36
    m.addConstrs(z[i, j] <= upper_b * y[i, j] for i in drivers for j in customers)
    #37
    m.addConstrs(cso[j] >= ro[i]*y[i,j] for i in drivers for j in customers)
    #38
    m.addConstrs(cso[j] <= (ro[i] - 1)*(1- y[i,j]) + driver_num*y[i,j] for i in drivers for j in customers)
    #39
    #m.addConstrs(b[i,j] == ro[i] for i in drivers for j in customers)
    m.addConstrs(gp.quicksum(b[i, j] for j in customers) == ro[i] for i in drivers)
    #40
    m.addConstrs(b[i,j] - cso[j] <= driver_num*(1 - x[i,j]) for i in drivers for j in customers)
    #41
    m.addConstrs(cso[j] - b[i, j]<= driver_num * (1 - x[i, j]) for i in drivers for j in customers)
    #42
    m.addConstrs(b[i, j] <= (driver_num)*x[i,j] for i in drivers for j in customers)

    #43
    m.addConstr(gp.quicksum(x[i, j] for i in drivers for j in rev_sp) >= req_sp_num)
    #44
    m.addConstrs(gp.quicksum(x[i, j] for j in customers) == 1 for i in drivers)
    #45
    m.addConstrs(gp.quicksum(x[i, j] for i in drivers) <= 1 for j in customers)
    #47
    m.addConstr(gp.quicksum(cso[j] for j in customers) == sum_i + (driver_num)*(customer_num - driver_num))
    #48
    m.addConstrs(cso[j] <= driver_num for j in customers)
    #m.addConstrs(cso[j] >= 1 for j in customers)
    for i in drivers:
        for j in customers:
            if lower_b != False:
                m.addConstr(lower_b <= v[i, j])
            if upper_b != False:
                m.addConstr(v[i, j] <= upper_b)
    if print_gurobi == False:
        m.setParam(GRB.Param.OutputFlag, 0)
    m.Params.method = solver  # -1은 auto dedection이며, 1~5에 대한 차이.
    m.optimize()
    res = printer(m.getVars(), [], len(drivers), len(customers))
    print('Obj val: %g' % m.objVal, "Solver", solver)
    c_list = []
    x_list = []
    for val in m.getVars():
        if val.VarName[0] == 'c':
            c_list.append(int(val.x))
        elif val.VarName[0] == 'x':
            x_list.append(int(val.x))
    #print("CSO")
    #print(c_list)
    c_list.sort()
    #print(c_list)
    x_list = np.array(x_list)
    x_list = x_list.reshape(driver_num, customer_num)
    #print("X")
    #print(x_list)
    return res, m.getVars()


def SimpleInverseSolverBasic(driver_set, customers_set, v_old, d_orders, times, end_times, lower_b = False, upper_b = False, sp=None,
                         print_gurobi=False, solver=-1, slack = 0, time_con = True):
    drivers = list(range(len(driver_set)))
    customers = list(range(len(customers_set)))
    """
    print('Problem input check')
    print('Dr orders', d_orders)
    print('Dr#', len(drivers),'Ct#', len(customers))
    print('v_old')
    print(v_old)
    """
    # D.V. and model set.
    m = gp.Model("mip1")
    x = m.addVars(len(drivers), len(customers), vtype=GRB.BINARY, name="x")
    v = m.addVars(len(drivers), len(customers), vtype=GRB.CONTINUOUS, name="v")  # GRB.INTEGER
    y = m.addVars(len(drivers), len(customers), vtype=GRB.BINARY, name="y")
    c_seq = m.addVars(len(customers), vtype=GRB.INTEGER, name="c")
    d = m.addVars(len(drivers), vtype=GRB.CONTINUOUS, name="d")
    if sp == None:
        num_sp = max(int(len(drivers) / 2), random.choice(drivers))
        sp = random.sample(customers, num_sp)
    req_sp_num = min(len(driver_set), len(sp))
    rev_sp = sp
    # Set objective
    m.setObjective(gp.quicksum(v[i, j] for i in drivers for j in customers), GRB.MINIMIZE)
    # constr 1. 사전에 할당된 고객 수행.
    m.addConstr(gp.quicksum(x[i, j] for i in drivers for j in rev_sp) >= req_sp_num)
    # constr 2 : 차량은 1명의 고객만 할당 가능.
    m.addConstrs(gp.quicksum(x[i, j] for j in customers) <= 1 for i in drivers)
    # constr 4: 고객은 1개의 차량에 의해서만 서비스 받을 수 있음.
    m.addConstrs(gp.quicksum(x[i, j] for i in drivers) <= 1 for j in customers)
    # 순서 관련 제약식
    m.addConstrs(
        (x[i, j] == 1) >> (d_orders[i] == c_seq[j]) for i in drivers for j in customers)  # <- indicator constraint
    # 변경된 제약식
    m.addConstr(
        gp.quicksum(c_seq[j] for j in customers) == sum(list(range(1, len(drivers) + 1))) + (len(drivers) + 1) * (
                    len(customers_set) - len(drivers)))
    m.addConstrs(c_seq[j] <= len(drivers) + 1 for j in customers)
    m.addConstrs((y[i, j] == 1) >> (d_orders[i] <= c_seq[j]) for i in drivers for j in customers)
    m.addConstrs((y[i, j] == 0) >> (d_orders[i] - 1 >= c_seq[j]) for i in drivers for j in customers)
    m.addConstrs((d[i] == gp.quicksum((v[i, j] + v_old[i, j]) * x[i, j] for j in customers)) for i in drivers)
    m.addConstrs((y[i, j] == 1) >> (d[i] >= v[i, j] + v_old[i, j] + slack) for i in drivers for j in customers)
    m.addConstrs(x[i, j] * (v[i, j] + v_old[i, j]) >= 0 for i in drivers for j in customers)
    if len(times) > 0 and time_con == True:
        m.addConstrs(x[i, j] * times[i, j] <= end_times[i, j] for i in drivers for j in customers)
    else:
        pass
        #m.addConstrs(x[i,j]*times[i,j] <= end_times[i,j] for i in drivers for j in customers)
    # Aconstr 7 : Add constraint: lower <= rev_v <= upeer value
    if lower_b != False:
        m.addConstrs(lower_b <= v[i, j] for i in drivers for j in customers)
    if upper_b != False:
        m.addConstrs(v[i, j] <= upper_b for i in drivers for j in customers)
    if print_gurobi == False:
        m.setParam(GRB.Param.OutputFlag, 0)
    m.Params.method = solver  # -1은 auto dedection이며, 1~5에 대한 차이.
    m.optimize()
    #print(d_orders)
    try:
        print('Obj val: %g' % m.objVal, "Solver", solver)
        res = printer(m.getVars(), [], len(drivers), len(customers))
        return res, m.getVars()
    except:
        print('Infeasible')
        #res = printer(m.getVars(), [], len(drivers), len(customers))
        return False, False


def SimpleInverseSolverBasicRelax(driver_set, customers_set, v_old, d_orders, times, end_times, lower_b = False, upper_b = False, sp=None,
                         print_gurobi=False, solver=-1, slack = 0, time_con = True, time_con_num = 0):
    drivers = list(range(len(driver_set)))
    customers = list(range(len(customers_set)))
    """
    print('Problem input check')
    print('Dr orders', d_orders)
    print('Dr#', len(drivers),'Ct#', len(customers))
    print('v_old')
    print(v_old)
    """
    # D.V. and model set.
    m = gp.Model("mip1")
    x = m.addVars(len(drivers), len(customers), vtype=GRB.BINARY, name="x")
    v = m.addVars(len(drivers), len(customers), vtype=GRB.CONTINUOUS, name="v")  # GRB.INTEGER
    y = m.addVars(len(drivers), len(customers), vtype=GRB.BINARY, name="y")
    c_seq = m.addVars(len(customers), vtype=GRB.INTEGER, name="c")
    d = m.addVars(len(drivers), vtype=GRB.CONTINUOUS, name="d")
    k = m.addVars(len(drivers), len(customers), vtype=GRB.BINARY, name="k")
    if sp == None:
        num_sp = max(int(len(drivers) / 2), random.choice(drivers))
        sp = random.sample(customers, num_sp)
    req_sp_num = min(len(driver_set), len(sp))
    rev_sp = sp
    # Set objective
    m.setObjective(gp.quicksum(v[i, j] for i in drivers for j in customers), GRB.MINIMIZE)
    # constr 1. 사전에 할당된 고객 수행.
    m.addConstr(gp.quicksum(x[i, j] for i in drivers for j in rev_sp) >= req_sp_num)
    # constr 2 : 차량은 1명의 고객만 할당 가능.
    m.addConstrs(gp.quicksum(x[i, j] for j in customers) <= 1 for i in drivers)
    # constr 4: 고객은 1개의 차량에 의해서만 서비스 받을 수 있음.
    m.addConstrs(gp.quicksum(x[i, j] for i in drivers) <= 1 for j in customers)
    # 순서 관련 제약식
    m.addConstrs(
        (x[i, j] == 1) >> (d_orders[i] == c_seq[j]) for i in drivers for j in customers)  # <- indicator constraint
    # 변경된 제약식
    m.addConstr(
        gp.quicksum(c_seq[j] for j in customers) == sum(list(range(1, len(drivers) + 1))) + (len(drivers) + 1) * (
                    len(customers_set) - len(drivers)))
    m.addConstrs(c_seq[j] <= len(drivers) + 1 for j in customers)
    m.addConstrs((y[i, j] == 1) >> (d_orders[i] <= c_seq[j]) for i in drivers for j in customers)
    m.addConstrs((y[i, j] == 0) >> (d_orders[i] - 1 >= c_seq[j]) for i in drivers for j in customers)
    m.addConstrs((d[i] == gp.quicksum((v[i, j] + v_old[i, j]) * x[i, j] for j in customers)) for i in drivers)
    m.addConstrs((y[i, j] == 1) >> (d[i] >= v[i, j] + v_old[i, j] + slack) for i in drivers for j in customers)
    m.addConstrs(x[i, j] * (v[i, j] + v_old[i, j]) >= 0 for i in drivers for j in customers)

    m.addConstrs((k[i, j] == 1) >> (x[i, j] * times[i, j] <= end_times[i, j]) for i in drivers for j in customers)
    m.addConstrs((k[i, j] == 0) >> (x[i, j] * times[i, j] >= end_times[i, j] + 0.0001) for i in drivers for j in customers)
    m.addConstr(gp.quicksum(k[i, j]*x[i, j] for i in drivers for j in customers) >= time_con_num)

    if len(times) > 0 and time_con == True:
        m.addConstrs(x[i, j] * times[i, j] <= end_times[i, j] for i in drivers for j in customers)
        pass
    else:
        pass
        #m.addConstrs(x[i,j]*times[i,j] <= end_times[i,j] for i in drivers for j in customers)
    # Aconstr 7 : Add constraint: lower <= rev_v <= upeer value
    if lower_b != False:
        m.addConstrs(lower_b <= v[i, j] for i in drivers for j in customers)
    if upper_b != False:
        m.addConstrs(v[i, j] <= upper_b for i in drivers for j in customers)
    if print_gurobi == False:
        m.setParam(GRB.Param.OutputFlag, 0)
    m.Params.method = solver  # -1은 auto dedection이며, 1~5에 대한 차이.
    m.optimize()
    #print(d_orders)
    try:
        print('Obj val: %g' % m.objVal, "Solver", solver)
        res = printer(m.getVars(), [], len(drivers), len(customers))
        return res, m.getVars()
    except:
        print('Infeasible')
        #res = printer(m.getVars(), [], len(drivers), len(customers))
        return False, False

def ReviseCoeffAP(selected, others, org_coeff):
    coeff = list(range(len(selected)))
    # D.V. and model set.
    m = gp.Model("mip1")
    x = m.addVars(len(selected), vtype=GRB.BINARY, name="x")
    z = m.addVar(type = GRB.CONTINUOUS, name= "z")
    m.setObjective(gp.quicksum(x[i] for i in coeff), GRB.MINIMIZE)
    m.addConstr(z == gp.quicksum((x[i]+org_coeff[i])*selected[i] for i in coeff))
    for other_info in others:
        m.addConstr(gp.quicksum((x[i]+org_coeff[i])*other_info[i] for i in coeff) <= z)
    m.optimize()
    #print(d_orders)
    try:
        print('Obj val: %g' % m.objVal)
        return True, m.getVars()
    except:
        print('Infeasible')
        return False, None



"""
driver_num = 10
customer_num = 30
driver_set = list(range(0,driver_num))
customers_set = list(range(0,customer_num))
orders = list(range(1,driver_num + 1))
random.shuffle(orders)
v_old = []
for i in range(driver_num*customer_num):
    v_old.append(random.randrange(5,25))
v_old = np.array(v_old)
v_old = v_old.reshape(driver_num,customer_num)
print(np.shape(v_old))
a,b =  SimpleInverseSolverBasic(driver_set, customers_set, v_old, orders,sp=None,print_gurobi=True)
"""
"""
#LinearizedSubsidyProblem(driver_set, customers_set, v_old, ro, lower_b = False, upper_b = False, sp=None, print_gurobi=False,  solver=-1)
driver_num = 20
customer_num = 70
driver_set = list(range(0,driver_num))
customers_set = list(range(0,customer_num))
orders = list(range(1,driver_num + 1))
random.shuffle(orders)
v_old = []
for i in range(driver_num*customer_num):
    v_old.append(random.randrange(5,25))
v_old = np.array(v_old)
v_old = v_old.reshape(driver_num,customer_num)
print(np.shape(v_old))
res , vars = LinearizedSubsidyProblem(driver_set, customers_set, v_old, orders, upper_b= 100,lower_b= 0 ,print_gurobi = True)
"""
"""
driver_set = list(range(5))
customers_set = list(range(10))
orders = list(range(1, len(driver_set) + 1))
print("orders1", orders)
random.shuffle(orders)
print("orders2", orders)
v_old_data = list(range(len(driver_set)*len(customers_set)))
random.shuffle(v_old_data)
v_old = np.array(v_old_data).reshape(len(driver_set),len(customers_set))
print(type(v_old), type(v_old) == np.ndarray)
x_old = SimpleAssingmentSolver(driver_set, customers_set , v_old)
"""
