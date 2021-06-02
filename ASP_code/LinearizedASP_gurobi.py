# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random
import subsidyASP as ASP

def LinearizedSubsidyProblem(driver_set, customers_set, v_old, ro, times, end_times, lower_b = False, upper_b = False, sp=None, print_gurobi=False,  solver=-1, delta = 100, relax = 100):
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
    sum_i = sum(ro)
    #print('parameters',drivers,customers, ro, driver_num, customer_num, sum_i)
    if upper_b == False:
        upper_b = 10000
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
    rev_sp = sp
    req_sp_num = min(len(driver_set), len(sp), relax)
    #print("Priority Customer", rev_sp)
    # Set objective #29
    m.setObjective(gp.quicksum(v[i, j] for i in drivers for j in customers), GRB.MINIMIZE)
    #32
    """
    for i in drivers:
        for j in customers:
            m.addConstr(gp.quicksum(w[i, k] + v_old[i, k] * x[i, k] for k in customers) >= z[i, j] + v_old[i, j] * y[i, j])
    """
    m.addConstrs(gp.quicksum(w[i,k] + v_old[i,k]*x[i,k] for k in customers) >= z[i,j] + v_old[i,j]*y[i,j] + delta for i in drivers for j in customers)
    #33
    m.addConstrs( w[i,j]-v[i,j ]<= upper_b*(1-x[i,j]) for i in drivers for j in customers)
    #34
    m.addConstrs(v[i, j] - w[i, j] <= upper_b*(1 - x[i, j]) for i in drivers for j in customers)
    #35
    m.addConstrs(w[i, j] <= upper_b * x[i, j] for i in drivers for j in customers)
    #36
    m.addConstrs(z[i, j] - v[i, j] <= upper_b*(1 - y[i, j]) for i in drivers for j in customers)
    #37
    m.addConstrs(v[i, j] - z[i, j] <= upper_b * (1 - y[i, j]) for i in drivers for j in customers)
    #38
    m.addConstrs(z[i, j] <= upper_b * y[i, j] for i in drivers for j in customers)
    #39
    m.addConstrs(cso[j] >= ro[i]*y[i,j] for i in drivers for j in customers)
    #40
    m.addConstrs(cso[j] <= (ro[i])*(1- y[i,j]) + driver_num*y[i,j] for i in drivers for j in customers)
    #41
    #m.addConstrs(b[i,j] == ro[i] for i in drivers for j in customers)
    m.addConstrs(gp.quicksum(b[i, j] for j in customers) == ro[i] for i in drivers)
    #42
    m.addConstrs(b[i,j] - cso[j] <= driver_num*(1 - x[i,j]) for i in drivers for j in customers)
    #43
    m.addConstrs(cso[j] - b[i, j]<= driver_num*(1 - x[i, j]) for i in drivers for j in customers)
    #44
    m.addConstrs(b[i, j] <= (driver_num)*x[i,j] for i in drivers for j in customers)
    #45
    m.addConstr(gp.quicksum(x[i, j] for i in drivers for j in rev_sp) >= req_sp_num)
    #46
    m.addConstrs(gp.quicksum(x[i, j] for j in customers) == 1 for i in drivers)
    #m.addConstrs(gp.quicksum(x[i, j] for j in customers) <= 1 for i in drivers)
    #47
    m.addConstrs(gp.quicksum(x[i, j] for i in drivers) <= 1 for j in customers)
    #49
    m.addConstr(gp.quicksum(cso[j] for j in customers) == sum_i + (driver_num)*(customer_num - driver_num))
    #50
    m.addConstrs(cso[j] <= driver_num for j in customers)
    #51 시간 제약식
    m.addConstrs(x[i,j]*times[i,j] <= end_times[i,j] for i in drivers for j in customers)
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
    """
    res = ASP.printer(m.getVars(), [], len(drivers), len(customers))
    print('Obj val: %g' % m.objVal, "Solver", solver)
    c_list = []
    x_list = []
    y_list = []
    for val in m.getVars():
        if val.VarName[0] == 'c':
            c_list.append(int(val.x))
        elif val.VarName[0] == 'x':
            x_list.append(int(val.x))
        elif val.VarName[0] == 'y':
            y_list.append(int(val.x))
        else:
            pass
    print("CSO")
    print(c_list)
    c_list.sort()
    print(c_list)
    x_list = np.array(x_list)
    x_list = x_list.reshape(driver_num, customer_num)
    print("X")
    print(x_list)
    print("Y")
    y_list = np.array(y_list)
    y_list = y_list.reshape(driver_num, customer_num)
    print(y_list)
    """
    try:
        print('Obj val: %g' % m.objVal, "Solver", solver)
        res = ASP.printer(m.getVars(), [], len(drivers), len(customers))
        return res, m.getVars()
    except:
        print('Infeasible')
        #res = printer(m.getVars(), [], len(drivers), len(customers))
        return False, False


def ReviseCoeffAP(selected, others, org_coeff, past_data = []):
    coeff = list(range(len(org_coeff)))
    # D.V. and model set.
    m = gp.Model("mip1")
    x = m.addVars(len(selected), vtype=GRB.CONTINUOUS, name="x")
    z = m.addVars(1 + len(past_data), vtype = GRB.CONTINUOUS, name= "z")
    m.setObjective(gp.quicksum(x[i] for i in coeff), GRB.MINIMIZE)
    z_count = 0
    #이번 selected와 other에 대한 문제 풀이
    m.addConstr(gp.quicksum((x[i] + org_coeff[i])*selected[i] for i in coeff) == z[z_count])
    for other_info in others:
        m.addConstr(gp.quicksum((x[i] + org_coeff[i])*other_info[i] for i in coeff) <= z[z_count] - 10)
    z_count += 1
    #과거 정보를 적층하는 작업
    if len(past_data) > 0:
        for data in past_data:
            p_selected = data[0]
            p_others = data[1]
            print('p_selected',p_selected)
            print('p_others',p_others)
            m.addConstr(gp.quicksum((x[i] + org_coeff[i]) * p_selected[i] for i in coeff) == z[z_count])
            for p_other_info in p_others:
                print('p_other_info',p_other_info)
                m.addConstr(gp.quicksum((x[i] + org_coeff[i]) * p_other_info[i] for i in coeff) <= z[z_count] - 10)
            z_count += 1


    #풀이
    m.optimize()
    try:
        print('Obj val: %g' % m.objVal)
        res = []
        for val in m.getVars():
            if val.VarName[0] == 'x':
                res.append(float(val.x))
        return True, res
    except:
        print('Infeasible')
        return False, None


"""
#LinearizedSubsidyProblem(driver_set, customers_set, v_old, ro, lower_b = False, upper_b = False, sp=None, print_gurobi=False,  solver=-1)
driver_num = 10
customer_num = 20
driver_set = list(range(1,driver_num + 1))
customers_set = list(range(1,customer_num + 1))
orders = list(range(1,driver_num + 1))
#orders = list(range(driver_num))
print('drivers',orders)
random.shuffle(orders)
v_old = []
for i in range(driver_num*customer_num):
    v_old.append(random.randrange(5,25))
v_old = np.array(v_old)
v_old = v_old.reshape(driver_num,customer_num)
print(np.shape(v_old))
times = np.zeros((driver_num,customer_num))
end_times = np.zeros((driver_num,customer_num))
#res , vars = LinearizedSubsidyProblem(driver_set, customers_set, v_old, orders,times, end_times, print_gurobi = True)
"""