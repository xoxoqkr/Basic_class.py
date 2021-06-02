# -*- coding: utf-8 -*-
import random
import math
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import ASP_code.ASP_class as asp
"""
import time
import simpy
import operator
from openpyxl import Workbook
from datetime import datetime
from sklearn.cluster import KMeans
import copy
"""

#인스턴스 생성
drivers = [0,1,2,3,4]
customers = [1,2,3,4,5,6,7,8,9,10]
dist_matrix = []
insert_cost_matrix = []
for i in drivers:
    tem1 = []
    tem2 = []
    for j in customers:
        tem1.append(random.randrange(9,15))
        tem2.append(random.randrange(5, 9))
    dist_matrix.append(tem1)
    insert_cost_matrix.append(tem2)

dist_matrix = np.array(dist_matrix).reshape(len(drivers), len(customers))
insert_cost_matrix = np.array(insert_cost_matrix).reshape(len(drivers), len(customers))
ava_CPs = [1,2,3,4,5]
return_cost = [20,30,40,50,25]
left_ct_time = []
for j in customers:
    left_ct_time.append(random.randrange(60,90))





def RouteInsertCost(req_route, customer, customer_set, name_para = False):
    route = [customer_set[0].location]
    for req in req_route:
        if name_para == False:
            route.append(customer_set[req.info].location)
        else:
            route.append(customer_set[req].location)
    if route[-1] != customer_set[0].location:
        route.append(customer_set[0].location)
    #print("Route check", route)
    res = []
    for index in list(range(1, len(route))):
        bf_n = route[index - 1]
        af_n = route[index]
        #print("RouteInsertCost",bf_n, af_n,customer)
        val = distance(bf_n, customer.location) + distance(customer.location, af_n) - distance(bf_n, af_n)
        res.append(val)
    return min(res)



def problem_setting(all_veh_set, customer_set, CP_set, now_time):
    print('problem_setting start')
    #veh_set = ReturnableVeh(all_veh_set, now_time)
    veh_set = ReturnableVeh2(all_veh_set, now_time)
    if len(veh_set) == 0:
        return None,None,None,None,None,None,None
    un_cts = asp.UnloadedCustomer(customer_set, now_time)
    veh_dist_matrix = []
    queue_len = []
    return_cost = []
    time_matrix = []
    remain_time = []
    un_cts_names = []
    for ct in un_cts:
        remain_time.append(ct.time_info[0] + ct.time_info[5] - now_time)
        un_cts_names.append(ct.name)
    print("ReturnableVeh",veh_set, "CTS", len(un_cts),un_cts[:4])
    for veh_index in veh_set:
        queue_len.append(len(all_veh_set[veh_index].veh.users) + len(all_veh_set[veh_index].veh.put_queue))
    max_queue_len = max(queue_len)
    for veh_index in veh_set:
        veh = all_veh_set[veh_index]
        if len(veh.veh.users) > 0:
            veh_tem_dist = []
            current_ct = customer_set[veh.veh.users[0].info]
            return_cost.append(2 * distance(customer_set[0].location, current_ct.location))
            for index in range(len(un_cts)):
                ct = un_cts[index]
                min_time = RouteInsertCost(veh.veh.users + veh.veh.put_queue, ct, customer_set)
                time_matrix.append(min_time)
                tem_dist = []
                for request in veh.veh.users + veh.veh.put_queue:
                    ct2 = customer_set[request.info]
                    tem_dist.append(distance(ct.location, ct2.location))
                    if request.info == veh.veh.users[0].info:
                        #return_cost.append(2 * distance(customer_set[0].location, ct2.location))
                        pass
                veh_tem_dist.append(min(tem_dist))
            veh_dist_matrix.append(veh_tem_dist)
        """
        else:
            veh_tem_dist = []
            for index in range(len(un_cts)):
                time_matrix.append(min_time)
        """

    veh_dist_matrix2 = np.array(veh_dist_matrix).reshape(len(veh_set), len(un_cts)) # min{삽입될 고객과 다른 고객 사이의 거리,..}
    ava_CPs = CP_sortor(CP_set)
    time_matrix2 = np.array(time_matrix).reshape(len(veh_set), len(un_cts)) #min{삽입될 고객으로 인해 경로가 증가하는 거리,..}
    return veh_dist_matrix2, ava_CPs, return_cost, time_matrix2, remain_time, veh_set, un_cts_names

def ReturnableVeh(veh_set, now_time):
    res = []
    for veh_index in veh_set:
        veh = veh_set[veh_index]
        print(veh.name, veh.call_back_info[-1])
        if veh.call_back_info[-1][2][2] == None:
            cond1 = False
        else:
            cond1 = veh.call_back_info[-1][2][1] < now_time - 60 # 회차가 수행된 시점이 1 시간 전이어야 함.
        print('cond1',cond1)
        cond2 = veh.call_back_info[-1][1] == True or veh.call_back_info[-1][0] == 'start'#처음 시작한 경우는 예외.
        cond3 = len(veh.veh.put_queue) > 0
        cond4 = True
        if (cond1 and cond2 == True) and (cond3 and cond4 == True):
            res.append(veh.name)
    return res

def ReturnableVeh2(veh_set, now_time):
    res = []
    for veh_index in veh_set:
        veh = veh_set[veh_index]
        print(veh.name, veh.call_back_info[-1])
        cond1 = veh.call_back_info[-1][2][0] < now_time - 60
        print(int(now_time),'cond1',cond1)
        cond2 = veh.call_back_info[-1][1] == True or veh.call_back_info[-1][0] == 'start'#처음 시작한 경우는 예외.
        cond3 = len(veh.veh.put_queue) > 0
        cond4 = True
        #input('ReturnableVeh2')
        print(cond1, cond2, cond3,cond4)
        if (cond1 and cond2 == True) and (cond3 and cond4 == True):
            res.append(veh.name)
    print(res)
    #input('ReturnableVeh2')
    return res



def indexReturner1D(list1D, val):
    res = []
    for index in range(len(list1D)):
        if int(list1D[index]) == val:
            res.append(index)
    return res

def indexReturner2D(list2D, val):
    res = []
    for row in range(len(list2D)):
        for index in range(len(list2D[row])):
            if int(list2D[row][index]) == val:
                res.append([row, index])
    return res

def indexReturner2DBiggerThan0(list2D, val):
    res = []
    for row in range(len(list2D)):
        for index in range(len(list2D[row])):
            if int(list2D[row][index]) > val:
                res.append([row, index, list2D[row][index]])
    return res

def solver_printer(getVars, num1, num2):
    res = []
    r_res = []
    x_res = []
    c_res = []
    d_res = []
    for val in getVars:
        res.append(abs(val.x))
        if  val.VarName[0] == 'r':
            r_res.append(val.x)
        elif val.VarName[0] == 'x':
            x_res.append(val.x)
        elif val.VarName[0] == 'c':
            c_res.append(val.x)
        elif val.VarName[0] == 'd':
            d_res.append(val.x)
    if len(x_res) > 0:
        x_res = np.array(x_res).reshape(num1, num2)
    print(x_res)
    print("r_res", r_res)
    print("c_res", c_res)
    print("d_res", d_res)
    print(r_res.index(1))
    print(c_res.index(1))
    return r_res, c_res, d_res

def solver_interpreter(getVars, num1, num2):
    res = []
    r_res = []
    x_res = []
    c_res = []
    d_res = []
    for val in getVars:
        try:
            res.append(abs(val.x))
            if  val.VarName[0] == 'r':
                r_res.append(int(val.x))
            elif val.VarName[0] == 'x':
                x_res.append(int(val.x))
            elif val.VarName[0] == 'c':
                c_res.append(int(val.x))
            elif val.VarName[0] == 'd':
                d_res.append(int(val.x))
        except:
            pass
    if len(x_res) > 0:
        x_res = np.array(x_res).reshape(num1, num2)
        #print("r_res", r_res)
        #print("x_res")
        try:
            print(x_res)
        except:
            pass
        r_index = indexReturner1D(r_res, 1)
        x_index = indexReturner2D(x_res, 1)
        c_index = indexReturner1D(c_res, 1)
        d_index = indexReturner1D(d_res, 1)
        return True, r_index, x_index, c_index, d_index
    else:
        return False, None, None, None, None

def CallBackDSM(env, r_index, x_index, c_index, d_index, veh_set, customer_set, now_time, driver_names, customer_names, driving_way = 'euc'):
    if len(r_index) == 0:
        return None
    returned = None
    for veh_index in r_index:
        veh = veh_set[driver_names[veh_index]]
        if len(veh.veh.put_queue) < 20 :
            print('callback done', round(env.now,2))
            veh.ToDepot(env, customer_set, return_type = 'callback' ,driving_way = driving_way)
            returned = True
        else:
            returned =  False
    for ct_index in c_index:
        ct = customer_set[customer_names[ct_index]]
        ct.Assign2CF_count += 1
        if ct.Assign2CF_count > 10:
            ct.Assign2CF = True
    print("r_index", r_index, "returned",returned, 'driver_names',driver_names)
    for info in x_index:
        print(info)
        if returned == True:
            customer_set[customer_names[info[1]]].arranged = driver_names[info[0]]
    print("come back veh",)
    for r in r_index:
        print(driver_names[r],)
    print("assigned cts",)
    for x in x_index:
        print(customer_names[x[1]],)
    print("CF assigned",)
    for c in c_index:
        print(customer_names[c],)
    return None



def Problem_solver_MO(dist_matrix, ava_CPs, veh_set, return_cost, left_ct_time = None, insert_cost_matrix = None,ava_veh = 2,turnOverCustomer = 2, slack = 2, cf_cost = 900, max_route_time = 90):
    drivers = list(range(len(veh_set)))
    customers = list(range(len(dist_matrix[0])))
    org_m = gp.Model("multiobj")
    #set parameter
    Subsets = range(2)
    SetObjPriority = [1, 1]
    SetObjWeight = [1.0, 1.0]
    revised_ava_CPs = min(0, len(ava_CPs) - slack)
    # Set global sense for ALL objectives
    org_m.ModelSense = GRB.MAXIMIZE
    # Limit how many solutions to collect
    org_m.setParam(GRB.Param.PoolSolutions, 1000)
    # Set number of objectives
    org_m.NumObj = 2
    x_co = []
    print("input size",len(drivers),len(customers),np.shape(return_cost),np.shape(dist_matrix))
    for i in range(len(drivers)):
        tem = []
        for j in range(len(customers)):
            tem.append(1)
        x_co.append(tem)
    x_co = np.array(x_co)
    # Create variables
    r = org_m.addVars(len(drivers), vtype=GRB.BINARY, name="r")
    x = org_m.addVars(len(drivers), len(customers), vtype=GRB.BINARY, name="x")
    c = org_m.addVars(len(customers), vtype=GRB.BINARY, name="c")
    d = org_m.addVars(len(customers), vtype=GRB.BINARY, name="d")
    sum_val = org_m.addVar(vtype=GRB.CONTINUOUS, name = "s")
    # obj1
    org_m.addConstr(sum_val >= gp.quicksum(return_cost[i] *r[i] for i in drivers)
                    + gp.quicksum(dist_matrix[i, j] * x[i, j] for i in drivers for j in customers)
                    + gp.quicksum(cf_cost*c[j] for j in customers))
    """
    org_m.setObjective(gp.quicksum(return_cost * r[i] for i in drivers)
                       + gp.quicksum(dist_matrix[i, j] * x[i, j] for i in drivers for j in customers)
                       + gp.quicksum(cf_cost * c[j] for j in customers), GRB.MINIMIZE)
    """
    #obj2
    #org_m.setObjective(gp.quicksum(x[i, j] for i in drivers for j in customers), GRB.MAXIMIZE)

    # Add constraint: 고객은 3 중 1개에 해당되어야 함.
    org_m.addConstrs(gp.quicksum(x[i,j] for i in drivers) + c[j] + d[j] == 1 for j in customers)
    #라우트의 증가하는 시간이 너무 많지 않아야 함.
    if type(insert_cost_matrix) == None:
        org_m.addConstrs(gp.quicksum(dist_matrix[i,j]*x[i,j] for j in customers) <= max_route_time for i in drivers)
    else:
        org_m.addConstrs(gp.quicksum(insert_cost_matrix[i, j] * x[i, j] for j in customers) - return_cost[i] <= max_route_time for i in drivers)
    org_m.addConstr(gp.quicksum(c[j] for j in customers) <= revised_ava_CPs)
    if ava_veh > 0:
        org_m.addConstr(gp.quicksum(r[i] for i in drivers) >= 1)
    org_m.addConstr(gp.quicksum(d[j] for j in customers) <= turnOverCustomer)
    org_m.addConstrs(r[i] >= x[i,j] for j in customers for i in drivers)
    if left_ct_time != None:
        #org_m.addConstrs((x[i, j] == 1) >> (return_cost[i] + insert_cost_matrix[i, j] <= left_ct_time[j] )for i in drivers for j in customers)
        org_m.addConstrs((return_cost[i] + insert_cost_matrix[i, j])*x[i , j] <= left_ct_time[j] for i in drivers for j in customers )
    for i in Subsets:
        org_m.setParam(GRB.Param.ObjNumber, i)
        org_m.ObjNPriority = SetObjPriority[i]
        org_m.ObjNWeight = SetObjWeight[i]
        #org_m.ObjNName = 'Set' + str(i)
        org_m.ObjNRelTol = 0.01
        org_m.ObjNAbsTol = 1.0 + i
        if i == 0:
            org_m.setAttr(GRB.Attr.ObjN, [sum_val], [-1])
        elif i == 1:
            org_m.setAttr(GRB.Attr.ObjN, x, x_co)
    org_m.optimize()
    #res = solver_printer(org_m.getVars(), len(drivers), len(customers))
    res = solver_interpreter(org_m.getVars(), len(drivers), len(customers))
    print("x", x)
    return res

def Problem_solver(dist_matrix, ava_CPs, veh_set, return_cost ,slack = 2, cf_cost = 900, max_route_time = 90):
    drivers = list(range(len(veh_set)))
    customers = list(range(len(dist_matrix[0])))
    # Create a new model
    org_m = gp.Model("mip1")
    # Create variables
    r = org_m.addVars(len(drivers), vtype=GRB.BINARY, name="r")
    x = org_m.addVars(len(drivers), len(customers), vtype=GRB.BINARY, name="x")
    c = org_m.addVars(len(customers), vtype=GRB.BINARY, name="c")
    d = org_m.addVars(len(customers), vtype=GRB.BINARY, name="d")
    rc = org_m.addVar(vtype=GRB.CONTINUOUS, name="rc")
    cf = org_m.addVar(vtype=GRB.CONTINUOUS, name="cf")
    ic = org_m.addVar(vtype=GRB.CONTINUOUS, name="ic")
    # obj1
    org_m.addConstr(rc >= gp.quicksum(return_cost * r[i] for i in drivers))
    org_m.addConstr(cf >= gp.quicksum(dist_matrix[i, j] * x[i, j] for i in drivers for j in customers))
    org_m.addConstr(ic >= gp.quicksum(cf_cost * c[j] for j in customers))
    org_m.setObjective(rc + cf + ic, GRB.MINIMIZE)
    """
    org_m.setObjective(gp.quicksum(return_cost * r[i] for i in drivers)
                       + gp.quicksum(dist_matrix[i, j] * x[i, j] for i in drivers for j in customers)
                       + gp.quicksum(cf_cost * c[j] for j in customers), GRB.MINIMIZE)
    """
    #obj2
    #org_m.setObjective(gp.quicksum(x[i, j] for i in drivers for j in customers), GRB.MAXIMIZE)

    # Add constraint: 고객은 3 중 1개에 해당되어야 함.
    org_m.addConstrs(gp.quicksum(x[i,j] for i in drivers) + c[j] + d[j] == 1 for j in customers)
    #라우트의 증가하는 시간이 너무 많지 않아야 함.
    org_m.addConstrs(gp.quicksum(dist_matrix[i,j]*x[i,j] for j in customers) <= max_route_time for i in drivers)
    org_m.addConstr(gp.quicksum(c[j] for j in customers) <= len(ava_CPs) - slack)
    org_m.addConstr(gp.quicksum(x[i, j] for i in drivers for j in customers) >= 3)
    org_m.addConstr(gp.quicksum(r[i] for i in drivers) >= 1)
    org_m.optimize()
    res = solver_printer(org_m.getVars(), len(drivers), len(customers))
    tem = []
    return res

def one_click_solver(env, driver_set, customer_set, CP_set, now_time, driving_way = None):
    dist_matrix, ava_CPs, return_cost, insert_cost_matrix, remain_time, veh_set, un_cts_names = problem_setting(driver_set, customer_set, CP_set, now_time)
    print('ASP info1', remain_time, veh_set,un_cts_names)
    if remain_time == None or len(veh_set) < 2 or len(un_cts_names) < 2:
        pass
    else:
        feaiblility, r_index, x_index, c_index, d_index = Problem_solver_MO(dist_matrix, ava_CPs, veh_set, return_cost, left_ct_time = remain_time, insert_cost_matrix = insert_cost_matrix)
        if feaiblility == True:
            print(x_index)
            CallBackDSM(env, r_index, x_index, c_index, d_index, driver_set, customer_set, now_time, veh_set, un_cts_names)
            input('feasible S')
    return None


def distance(x1, x2, dist_cal_type='euc'):
    if dist_cal_type == 'euc':
        return round(math.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2), 2)
    else:
        return round(abs(x1[0] - x2[0]) + abs(x1[1] - x2[1]), 2)


def CP_sortor(CP_set):
    ava_CPs = []
    for cp_name in CP_set:
        cp_cond1 = CP_set[cp_name].left == False
        cp_cond2 = CP_set[cp_name].Done[0] == False
        if cp_cond1 == True and cp_cond2 == True:
            ava_CPs.append(cp_name)
    return ava_CPs
#print(dist_matrix)
#Problem_solver_MO(dist_matrix, ava_CPs, drivers, return_cost, left_ct_time = left_ct_time, insert_cost_matrix = insert_cost_matrix)

#Problem_solver(dist_matrix, ava_CPs, drivers, return_cost)