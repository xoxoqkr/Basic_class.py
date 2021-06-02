# -*- coding: utf-8 -*-
import numpy as np
import random
from docplex.mp.model import Model

#from docplex.mp.constr import *
from docplex.mp.constr import (LinearConstraint as DocplexLinearConstraint,
                               QuadraticConstraint as DocplexQuadraticConstraint,
                               NotEqualConstraint, IfThenConstraint)

driver_set = list(range(5))
customer_set = list(range(10))
v_old1 = []
for i in range(50):
    v_old1.append(random.randrange(5,20))
v_old = np.array(v_old1).reshape(5,10)
d_orders = [1,3,4,2,5]
required_cts = [1,4,7,9]


def cplex_inverse_solver(driver_set, customers_set, v_old, required_cts, lower_b = 10, upper_b = 100):

    drivers = list(range(len(driver_set)))
    customers = list(range(len(customers_set)))
    md1 = Model('IP_model')


    x = md1.binary_var_matrix(len(drivers), len(customers),name = 'x')
    v = md1.continuous_var_matrix(len(drivers), len(customers),name = 'v')
    z = md1.continuous_var_list(len(drivers) ,name = 'z')
    abs_v = md1.continuous_var_matrix(len(drivers), len(customers),name = 'abs_v')

    # lower & upper bound
    md1.add_constraints(
        (v[driver, customer] + v_old[driver,customer] >= lower_b) for driver in drivers for customer in customers)
    md1.add_constraints(
        (v[driver, customer] + v_old[driver,customer] <= upper_b) for driver in drivers for customer in customers)
    #assigment const
    #for customer in customers:
    #    md1.add_constraint(md1.sum(x[driver, customer] for driver in driver_set) == 1)
    md1.add_constraints(md1.sum(x[driver, customer] for driver in driver_set) <= 1 for customer in customers)
    md1.add_constraints(md1.sum(x[driver, customer] for customer in customers) == 1 for driver in driver_set)
    #필요한 고객 할당
    md1.add_constraint(md1.sum(x[driver, customer] for driver in drivers for customer in required_cts) >= len(required_cts))
    #선택되는 고객의 가치가 드라이버가 선택할 수 있었던 고객들의 가치 중 최대가 되도록
    #for driver in drivers:
    #    md1.add_constraint(z[driver] >= md1.max(v[driver, customer] for customer in customers))
    md1.add_constraints(z[driver] >= md1.max(v[driver, customer] + v_old[driver, customer] for customer in customers) for driver in drivers)
    #abs_v 정의
    md1.add_constraints(abs_v[driver, customer] == md1.abs(v[driver, customer]) for driver in drivers for customer in customers)
    #OBJ
    #md1.minimize(md1.sum(v[driver, customer] for driver in drivers for customer in customers))
    md1.minimize(md1.sum(abs_v[driver, customer] for driver in drivers for customer in customers))
    msol = md1.solve()
    return msol


def cplex_inverse_seq_solver(driver_set, customers_set, v_old, required_cts, d_orders, lower_b = 10, upper_b = 100):
    #input 변수 정의
    drivers = list(range(len(driver_set)))
    customers = list(range(len(customers_set)))
    mdseq = Model(name='qpex1')
    #D.V. 정의
    x = mdseq.binary_var_matrix(len(drivers), len(customers),name = 'x')
    v = mdseq.continuous_var_matrix(len(drivers), len(customers),name = 'v')
    c_seq = mdseq.integer_var_list(len(customers), name = 'c_seq')
    y = mdseq.binary_var_matrix(len(drivers), len(customers),name = 'y')
    d = mdseq.continuous_var_list(len(drivers) ,name = 'd')
    #제약식
    # lower & upper bound
    mdseq.add_constraints(
        (v[driver, customer] + v_old[driver,customer] >= lower_b) for driver in drivers for customer in customers)
    mdseq.add_constraints(
        (v[driver, customer] + v_old[driver,customer] <= upper_b) for driver in drivers for customer in customers)
    #assigment const
    mdseq.add_constraints(mdseq.sum(x[driver, customer] for driver in driver_set) <= 1 for customer in customers)
    mdseq.add_constraints(mdseq.sum(x[driver, customer] for customer in customers) == 1 for driver in driver_set)
    #필요한 고객 할당
    mdseq.add_constraint(mdseq.sum(x[driver, customer] for driver in drivers for customer in required_cts) >= len(required_cts))
    #c_seq 제약식
    mdseq.add_constraint(mdseq.sum(c_seq[customer] for customer in customers) == sum(list(range(1, len(drivers) + 1)))
                       + (len(customers) - len(drivers))*(len(drivers) + 1))
    mdseq.add_constraints(c_seq[customer] <= len(drivers) + 1 for customer in customers)

    for driver in drivers:
        for customer in customers:
            mdseq.add_if_then(x[driver, customer] == 1, d[driver] == v[driver, customer] + v_old[driver, customer]) #순서 제약식
            mdseq.add_if_then(y[driver, customer] == 1, d[driver] >= v[driver, customer] + v_old[driver, customer]) #순서 제약식
            mdseq.add_if_then(y[driver, customer] == 0, d_orders[driver] - 1 >= c_seq[customer]) #순서 제약식
            mdseq.add_if_then(y[driver, customer] == 1, d_orders[driver] <= c_seq[customer]) #순서 제약식
            mdseq.add_if_then(x[driver, customer] == 1, d_orders[driver] == c_seq[customer]) #synchromization

    #OBJ
    mdseq.minimize(mdseq.sum(v[driver, customer] for driver in drivers for customer in customers))
    msol = mdseq.solve()
    return msol


def cplex_inverse_seq_solver_tem(driver_set, customers_set, v_old, required_cts, d_orders, lower_b = 10, upper_b = 100):

    drivers = list(range(len(driver_set)))
    customers = list(range(len(customers_set)))
    mdseq = Model(name='qpex1')


    x = mdseq.binary_var_matrix(len(drivers), len(customers),name = 'x')
    v = mdseq.continuous_var_matrix(len(drivers), len(customers),name = 'v')
    c_seq = mdseq.integer_var_list(len(customers), name = 'c_seq')
    y = mdseq.binary_var_matrix(len(drivers), len(customers),name = 'y')
    d = mdseq.continuous_var_list(len(drivers) ,name = 'd')

    # lower & upper bound
    mdseq.add_constraints(
        (v[driver, customer] + v_old[driver,customer] >= lower_b) for driver in drivers for customer in customers)
    mdseq.add_constraints(
        (v[driver, customer] + v_old[driver,customer] <= upper_b) for driver in drivers for customer in customers)
    #assigment const
    mdseq.add_constraints(mdseq.sum(x[driver, customer] for driver in driver_set) <= 1 for customer in customers)
    mdseq.add_constraints(mdseq.sum(x[driver, customer] for customer in customers) == 1 for driver in driver_set)
    #필요한 고객 할당
    mdseq.add_constraint(mdseq.sum(x[driver, customer] for driver in drivers for customer in required_cts) >= len(required_cts))
    #선택되는 고객의 가치가 드라이버가 선택할 수 있었던 고객들의 가치 중 최대가 되도록
    #mdseq.add_constraints(z[driver] >= mdseq.max(v[driver, customer] + v_old[driver, customer] for customer in customers) for driver in drivers)
    #고객 순서 제약식
    #mdseq.add_constraints(d_orders[driver] == c_seq[customer] for driver in drivers for customer in customers
    #                      if x[driver, customer] is 1)
    #mdseq.add_constraints((x[driver, customer] is 1) >> d_orders[driver] == c_seq[customer] for driver in drivers for customer in customers)
    mdseq.add_constraint(mdseq.sum(c_seq[customer] for customer in customers) == sum(list(range(1, len(drivers) + 1)))
                       + (len(customers) - len(drivers))*(len(drivers) + 1))
    print("c_seq sum", sum(list(range(1, len(drivers) + 1))) + (len(customers) - len(drivers))*(len(drivers) + 1))
    mdseq.add_constraints(c_seq[customer] <= len(drivers) + 1 for customer in customers)
    #mdseq.add_constraints((y[driver, customer] is 1) >> d_orders[driver] <= c_seq[customer] for driver in drivers for customer in customers)

    #mdseq.add_constraints(d_orders[driver] <= c_seq[customer] for driver in drivers for customer in customers
    #                    if y[driver, customer] is 1)
    #mdseq.if_then(if y[driver, customer] == 1, d_orders[driver] <= c_seq[customer])
    #IfThenConstraint(mdseq, y[driver, customer] == 1, d_orders[driver] <= c_seq[customer])
    #mdseq.add_constraints((y[driver, customer] is 0) >> d_orders[driver] - 1 >= c_seq[customer] for driver in drivers for customer in customers)
    #mdseq.add_constraints(d_orders[driver] - 1 >= c_seq[customer] for driver in drivers for customer in customers
    #                    if y[driver, customer] is 0) #docplex.mp.QuadraticConstraint
    #mdseq.DocplexQuadraticConstraint(d[driver] == mdseq.sum((v[driver, customer] + v_old[driver, customer])*x[driver, customer] for customer in customers)
    #                    for driver in drivers)
    #mdseq.add_constraints((x[driver, customer] is 1) >> d[driver] == v[driver, customer] + v_old[driver, customer] for driver in drivers for customer in customers
    #                     )
    for driver in drivers:
        for customer in customers:
            mdseq.add_if_then(x[driver, customer] == 1, d[driver] == v[driver, customer] + v_old[driver, customer])
            mdseq.add_if_then(y[driver, customer] == 1, d[driver] >= v[driver, customer] + v_old[driver, customer])
            mdseq.add_if_then(y[driver, customer] == 0, d_orders[driver] - 1 >= c_seq[customer])
            mdseq.add_if_then(y[driver, customer] == 1, d_orders[driver] <= c_seq[customer])
            mdseq.add_if_then(x[driver, customer] == 1, d_orders[driver] == c_seq[customer])
    #mdseq.add_constraints(d[driver] == v[driver, customer] + v_old[driver, customer] for driver in drivers for customer in customers
    #                     if x[driver, customer] is 1)
    #mdseq.add_constraints((y[driver, customer] is 1) >> d[driver] >= v[driver, customer] + v_old[driver, customer] for driver in drivers for customer in customers
    #                      )

    #mdseq.add_constraints(d[driver] >= v[driver, customer] + v_old[driver, customer] for driver in drivers for customer in customers
    #                      if y[driver, customer] is 1)
    #OBJ
    #mdseq.minimize(mdseq.sum(abs_v[driver, customer] for driver in drivers for customer in customers))
    mdseq.minimize(mdseq.sum(v[driver, customer] for driver in drivers for customer in customers))
    msol = mdseq.solve()
    return msol

def cplex_inverse_seq_solver_tem(driver_set, customers_set, v_old, required_cts, d_orders, lower_b = 10, upper_b = 100):

    drivers = list(range(len(driver_set)))
    customers = list(range(len(customers_set)))
    mdseq = Model(name='qpex1')


    x = mdseq.binary_var_matrix(len(drivers), len(customers),name = 'x')
    v = mdseq.continuous_var_matrix(len(drivers), len(customers),name = 'v')
    c_seq = mdseq.integer_var_list(len(customers), name = 'c_seq')
    y = mdseq.binary_var_matrix(len(drivers), len(customers),name = 'y')
    d = mdseq.continuous_var_list(len(drivers) ,name = 'd')

    # lower & upper bound
    mdseq.add_constraints(
        (v[driver, customer] + v_old[driver,customer] >= lower_b) for driver in drivers for customer in customers)
    mdseq.add_constraints(
        (v[driver, customer] + v_old[driver,customer] <= upper_b) for driver in drivers for customer in customers)
    #assigment const
    mdseq.add_constraints(mdseq.sum(x[driver, customer] for driver in driver_set) <= 1 for customer in customers)
    mdseq.add_constraints(mdseq.sum(x[driver, customer] for customer in customers) == 1 for driver in driver_set)
    #필요한 고객 할당
    mdseq.add_constraint(mdseq.sum(x[driver, customer] for driver in drivers for customer in required_cts) >= len(required_cts))
    #선택되는 고객의 가치가 드라이버가 선택할 수 있었던 고객들의 가치 중 최대가 되도록
    #mdseq.add_constraints(z[driver] >= mdseq.max(v[driver, customer] + v_old[driver, customer] for customer in customers) for driver in drivers)
    #고객 순서 제약식
    #mdseq.add_constraints(d_orders[driver] == c_seq[customer] for driver in drivers for customer in customers
    #                      if x[driver, customer] is 1)
    #mdseq.add_constraints((x[driver, customer] is 1) >> d_orders[driver] == c_seq[customer] for driver in drivers for customer in customers)
    mdseq.add_constraint(mdseq.sum(c_seq[customer] for customer in customers) == sum(list(range(1, len(drivers) + 1)))
                       + (len(customers) - len(drivers))*(len(drivers) + 1))
    print("c_seq sum", sum(list(range(1, len(drivers) + 1))) + (len(customers) - len(drivers))*(len(drivers) + 1))
    mdseq.add_constraints(c_seq[customer] <= len(drivers) + 1 for customer in customers)
    #mdseq.add_constraints((y[driver, customer] is 1) >> d_orders[driver] <= c_seq[customer] for driver in drivers for customer in customers)

    #mdseq.add_constraints(d_orders[driver] <= c_seq[customer] for driver in drivers for customer in customers
    #                    if y[driver, customer] is 1)
    #mdseq.if_then(if y[driver, customer] == 1, d_orders[driver] <= c_seq[customer])
    #IfThenConstraint(mdseq, y[driver, customer] == 1, d_orders[driver] <= c_seq[customer])
    #mdseq.add_constraints((y[driver, customer] is 0) >> d_orders[driver] - 1 >= c_seq[customer] for driver in drivers for customer in customers)
    #mdseq.add_constraints(d_orders[driver] - 1 >= c_seq[customer] for driver in drivers for customer in customers
    #                    if y[driver, customer] is 0) #docplex.mp.QuadraticConstraint
    #mdseq.DocplexQuadraticConstraint(d[driver] == mdseq.sum((v[driver, customer] + v_old[driver, customer])*x[driver, customer] for customer in customers)
    #                    for driver in drivers)
    #mdseq.add_constraints((x[driver, customer] is 1) >> d[driver] == v[driver, customer] + v_old[driver, customer] for driver in drivers for customer in customers
    #                     )
    for driver in drivers:
        for customer in customers:
            mdseq.add_if_then(x[driver, customer] == 1, d[driver] == v[driver, customer] + v_old[driver, customer])
            mdseq.add_if_then(y[driver, customer] == 1, d[driver] >= v[driver, customer] + v_old[driver, customer])
            mdseq.add_if_then(y[driver, customer] == 0, d_orders[driver] - 1 >= c_seq[customer])
            mdseq.add_if_then(y[driver, customer] == 1, d_orders[driver] <= c_seq[customer])
            mdseq.add_if_then(x[driver, customer] == 1, d_orders[driver] == c_seq[customer])
    #mdseq.add_constraints(d[driver] == v[driver, customer] + v_old[driver, customer] for driver in drivers for customer in customers
    #                     if x[driver, customer] is 1)
    #mdseq.add_constraints((y[driver, customer] is 1) >> d[driver] >= v[driver, customer] + v_old[driver, customer] for driver in drivers for customer in customers
    #                      )

    #mdseq.add_constraints(d[driver] >= v[driver, customer] + v_old[driver, customer] for driver in drivers for customer in customers
    #                      if y[driver, customer] is 1)
    #OBJ
    #mdseq.minimize(mdseq.sum(abs_v[driver, customer] for driver in drivers for customer in customers))
    mdseq.minimize(mdseq.sum(v[driver, customer] for driver in drivers for customer in customers))
    msol = mdseq.solve()
    return msol

def cplex_linearization_solver(driver_set, customers_set, v_old, ro, sp = None , lower_b = False, upper_b = False):
    drivers = list(range(len(driver_set)))
    customers = list(range(len(customers_set)))
    driver_num = len(driver_set)
    customer_num = len(customers_set)
    #sum_i = sum(ro)
    sum_i = sum(list(range(driver_num)))
    if upper_b == False:
        upper_b = 1000
    if lower_b == False:
        lower_b = 0
    #우선 고객 할당.
    if sp == None:
        num_sp = max(int(len(drivers) / 2), random.choice(drivers))
        sp = random.sample(customers, num_sp)
    req_sp_num = min(len(driver_set), len(sp))
    rev_sp = sp
    print("Priority Customer", rev_sp)
    mdseq = Model(name='qpex1')
    # D.V. and model set.
    x = mdseq.binary_var_matrix(len(drivers), len(customers),name = 'x')
    v = mdseq.continuous_var_matrix(len(drivers), len(customers), name='v')
    cso = mdseq.integer_var_list(len(customers), name='c')
    # 선형화를 위한 변수
    y = mdseq.binary_var_matrix(len(drivers), len(customers), name='y')
    w = mdseq.continuous_var_matrix(len(drivers), len(customers), name='w')
    z = mdseq.continuous_var_matrix(len(drivers), len(customers), name='z')
    b = mdseq.continuous_var_matrix(len(drivers), len(customers), name='b')

    #30
    for i in drivers:
        for j in customers:
            mdseq.add_constraint(mdseq.sum(w[i, k] + v[i,k]*x[i,k] for k in customers) >= z[i,j] + v[i,j]*y[i,j])
    #mdseq.add_constraints(mdseq.sum(w[i, k] + v[i,k]*x[i,k] for k in customers) >= z[i,j] + v[i,j]*y[i,j] for i in drivers for j in customers)
    #docplex.mp.QuadraticConstraint[]
    #mdseq.mp.QuadraticConstraint(mdseq.sum(w[i, k] + v[i,k]*x[i,k] for k in customers) >= z[i,j] + v[i,j]*y[i,j] for i in drivers for j in customers)
    #31
    mdseq.add_constraints(w[i,j] - v[i,j] <= upper_b*(1 - x[i,j]) for i in drivers for j in customers)
    #32
    mdseq.add_constraints(v[i, j] - w[i, j] <= upper_b * (1 - x[i, j]) for i in drivers for j in customers)
    #33
    mdseq.add_constraints(w[i,j] <= upper_b * x[i, j] for i in drivers for j in customers)
    #34
    mdseq.add_constraints(z[i, j] - v[i, j] <= upper_b * (1 - y[i, j]) for i in drivers for j in customers)
    #35
    mdseq.add_constraints(v[i, j] - z[i, j] <= upper_b * (1 - y[i, j]) for i in drivers for j in customers)
    #36
    mdseq.add_constraints(z[i, j] <= upper_b * y[i, j] for i in drivers for j in customers)
    #37
    mdseq.add_constraints(cso[j] >= ro[i]*y[i,j] for i in drivers for j in customers)
    #38
    mdseq.add_constraints(cso[j] <= (ro[i]-1)*(1-y[i,j]) + driver_num*y[i,j] for i in drivers for j in customers)
    #39
    mdseq.add_constraints(mdseq.sum(b[i,j] for j in customers) == ro[i]for i in drivers)
    #40
    mdseq.add_constraints(b[i,j] - cso[j] <= driver_num*(1-x[i,j]) for i in drivers for j in customers)
    #41
    mdseq.add_constraints(cso[j] - b[i, j]<= driver_num * (1 - x[i, j]) for i in drivers for j in customers)
    #42
    mdseq.add_constraints(b[i, j] <= driver_num*x[i,j] for i in drivers for j in customers)
    #43
    mdseq.add_constraint(mdseq.sum(x[i,j] for i in drivers for j in sp) >= req_sp_num)
    #44
    mdseq.add_constraints(mdseq.sum(x[i,j] for j in customers) == 1 for i in drivers)
    #45
    mdseq.add_constraints(mdseq.sum(x[i, j] for i in drivers) <= 1 for j in customers)
    #47
    mdseq.add_constraint(mdseq.sum(cso[j] for j in customers) == sum_i + driver_num*(customer_num-driver_num))
    #48
    mdseq.add_constraints(cso[j] <= driver_num for j in customers)
    #29
    mdseq.minimize(mdseq.sum(v[driver, customer] for driver in drivers for customer in customers))
    msol = mdseq.solve()
    return msol





"""
res1 = cplex_inverse_solver(driver_set, customer_set, v_old, required_cts)
print("solution1")
print(res1)
res2 = cplex_inverse_seq_solver_tem(driver_set, customer_set, v_old, required_cts, d_orders)

res3 = cplex_linearization_solver()
print("solution2")
print(res2)
"""
driver_num = 10
customer_num = 20
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
res3 = cplex_linearization_solver(driver_set,customers_set,v_old,orders, upper_b= 100,lower_b= 0)

