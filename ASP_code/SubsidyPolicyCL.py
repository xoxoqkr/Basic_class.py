# -*- coding: utf-8 -*-
import operator
import numpy as np
from ASP_code import AssignPSolver
import LinearizedASP_gurobi as lpg
import Basic_class as Basic


def FeeUpdater(rev_v, customer_set, riders, rider_set ,cts_name, now_time, subsidy_offer = [],subsidy_offer_count= [], upper = 10000, toCenter = 'Back_to_center'):
    """
    구한 보조금을 고객에게 반영
    고객들의 fees를 update.
    :param rev_v:
    :param customer_set:
    :param riders:
    :param rider_set:
    :param cts_name:
    :param now_time:
    :return:
    """
    for info in rev_v:
        try:
            rider_name = riders[info[0]]
            rider = rider_set[rider_name]
            customer = customer_set[cts_name[info[1]]]
            if info[2] < upper and customer.name > 0:
                customer.fee[1] = info[2] + 10
                customer.fee[2] = rider.name
                customer.fee[3] = now_time
                subsidy_offer.append([rider.name, customer.name])
                print('Fee offer to rider#',rider.name,'//end T:', round(rider.end_time,2),'//Rider left t',round(rider.gen_time + 120,2),'//CT #',customer.name,'//CT end T', customer.time_info[0] + customer.time_info[5],'$',customer.fee[0] + customer.fee[1], '//offered$', customer.fee[1])
                #tem = Basic.PriorityOrdering(rider, Basic.UnloadedCustomer(customer_set, now_time),now_time, toCenter = 'Back_to_center')
                subsidy_offer_count[int(now_time//60)] += 1
                #print(tem)
            else:
                pass
        except:
            print('Dummy customer')
            pass
    return True


def solver_interpreterForIP2(res_v):
    """
    IP가 푼 문제의 해의 feasilbility를 출력하고, 제안된 보조금(v)를 반환
    :param res_x:
    :param res_v:
    :param v_old:
    :return:
    """
    if type(res_v) == list:
        v = [[0,0,res_v[0]]]
    else:
        v = AssignPSolver.indexReturner2DBiggerThan0(res_v, 1)
    print('check v',v)
    if len(v) > 0:
        return True, v
    else:
        return False, None


def ProblemInput(rider_set, customer_set, now_time, minus_para = False, add = False, upper = 1500, std_para = False, mean = 0,std = 1.0, toCenter = 'Back_to_center'):
    print('Problem input')
    riders = Basic.AvaRider(rider_set, now_time)
    cts = Basic.UnloadedCustomer(customer_set, now_time)
    values = []
    d_orders = []
    times = []
    end_times = []
    #std_pool = np.random.normal(mean, std, 1000)
    for rider_name in riders:
        rider = rider_set[rider_name]
        d_orders.append([rider_name, rider.end_time])
        #error = np.random.choice(std_pool)
        for customer in cts:
            #time = CalTime(rider.last_location, rider.speed, customer)
            #time = CalTime2(rider.last_location, rider.speed, customer, center = customer.location[1])
            time = Basic.CalTime2(rider.exp_last_location, rider.speed, customer, center=[36,36], toCenter = toCenter, customer_set = cts)
            cost = (time / 60) * rider.wageForHr
            #print('Cost Cal',customer.name,customer.fee[0] + customer.fee[1], cost)
            paid = 0
            if customer.fee[2] == rider.name:
                paid += customer.fee[1]
            value = round(customer.fee[0] + paid - cost,2)
            #value += rider.error

            #error = np.random.choice(std_pool)
            if std_para == True:
                #value += value*error
                value += rider.error

            times.append(now_time + time)
            end_times.append(customer.time_info[0] + customer.time_info[5])
            #print('IF served',round(now_time + time,2), '/End Time:',round(customer.time_info[0] + customer.time_info[5],2),'/', rider.name,'->' ,customer.name)
            if minus_para == False:
                if value > 0 and rider.end_time + time < customer.time_info[0] + customer.time_info[5]:
                    values.append(value)
                else:
                    values.append(0)
            else:
                values.append(value)
    v_old = np.array(values).reshape(len(riders), len(cts))
    times = np.array(times).reshape(len(riders), len(cts))
    end_times = np.array(end_times).reshape(len(riders), len(cts))
    if add == False:
        if len(riders) > len(cts):
            #print('need more cts')
            dif = len(riders) - len(cts)
            add_array1 = np.zeros((len(riders),dif))
            v_old = np.hstack((v_old, add_array1))
            add_array2 = np.zeros((len(riders),dif))
            add_array3 = np.zeros((len(riders),dif))
            for i in range(0,len(add_array2)):
                for j in range(0,len(add_array2[i])):
                    add_array2[i,j] = 1000
            times = np.hstack((times, add_array3))
            end_times = np.hstack((end_times, add_array2))
        #print(v_old)
    #v_old = np.array(values).reshape(len(riders), len(cts))
    #times = np.array(times).reshape(len(riders), len(cts))
    #end_times= np.array(end_times).reshape(len(riders), len(cts))
    elif len(riders) > 1:
        add_array4 = np.zeros((len(riders), len(riders)))
        for index in range(0,len(add_array4[0])):
            for row in range(0,len(add_array4)):
                add_array4[row, index] = -upper
        v_old = np.hstack((v_old, add_array4))
        add_array5 = np.zeros((len(riders), len(riders)))
        add_array6 = np.zeros((len(riders), len(riders)))
        for i in range(0,len(add_array5)):
            for j in range(0,len(add_array5[i])):
                add_array5[i,j] = 1000
        times = np.hstack((times, add_array6))
        end_times = np.hstack((end_times, add_array5))
    else:
        pass
    cts_name = []
    for ct in cts:
        cts_name.append(ct.name)
    if len(riders) > len(cts):
        dif = len(end_times[0])-len(cts)
        for i in range(0,dif):
            cts_name.append(0)
    #d_orders를 계산
    d_orders.sort(key = operator.itemgetter(1))
    d_orders_res = [] #todo : 현재는 라우트의 길이가 짧을 수록 고객을 더 먼저 선택할 것이라고 가정함.
    for rider_name in riders:
        index = 1
        for info in d_orders:
            if rider_name == info[0]:
                d_orders_res.append(index)
                break
            index += 1
    print('d_orders_res',d_orders_res)
    return v_old, riders, cts_name, d_orders_res, times, end_times

def ExpectedSCustomer(rider_set, rider_names, d_orders_res, customer_set, now_time, toCenter = 'Back_to_center', who = 'platform'):
    """
    d_orders_res 순서로 rider가 주문을 선택한다고 하였을 때, 선택될 고객들을 반환
    :param rider_set: 라이더 dict
    :param rider_names: 주문을 수행할 수 있는 라이더 이름
    :param d_orders_res: 라이더 주문 수행 순서 rider_names의
    :param customer_set:
    :param now_time:
    :param toCenter:
    :return:
    """
    #print('rider_names', rider_names)
    #print('d_orders_res',d_orders_res, sorted(d_orders_res))
    #input('check')
    expected_cts = []
    customers = Basic.UnloadedCustomer(customer_set, now_time)
    already_selected = []  # 이미 선택되었을 고객.
    add_info = []
    for index in sorted(d_orders_res):
        test = []
        rider_name = rider_names[d_orders_res.index(index)]
        #print('rider_name',rider_name)
        rider = rider_set[rider_name]
        #rider = rider_set[rider_names[rider_name - 1]]
        future_last = customer_set[rider.now_ct].location[1] #현재 수행하고 있는 고객의 last location으로 부터의 거리를 계산해야 함.
        ct_infos = Basic.PriorityOrdering(rider, customers, now_time=now_time, toCenter=toCenter, who=who, last_location=future_last)
        #print(ct_infos)
        for info in ct_infos:
            if info[0] not in already_selected and info[1] > 0:
                expected_cts.append(info[0])
                already_selected.append(info[0])
                add_info.append([rider_name, info[0]])
                test.append([rider_name, ct_infos])
                customers.remove(customer_set[info[0]])
                break
        print('라이더 예상 고객은?',rider_name ,"::",test)
    return expected_cts, add_info

def SystemRunner(env, rider_set, customer_set, cool_time, interval=10, No_subsidy=False, subsidy_offer=[],
                 subsidy_offer_count=[], peak_times=[[0, 900]], time_thres=0.8, upper=10000, add_para=False,
                 checker=False, std_para=False, mean=0, std=1.0, toCenter='Back_to_center'):
    while env.now <= cool_time:
        # 보조금 문제를 풀 필요가 없는 경우에는 문제를 풀지 않아야 한다.
        # C_p에 해당하는 고객이 이미 선택될 것으로 예상되는 경우
        # 라이더 체크 확인
        ava_rider_names = Basic.AvaRider(rider_set, env.now)  # 가능한 라이더를 식별.
        un_cts = Basic.UnloadedCustomer(customer_set, env.now)  # 아직 실리지 않은 고객 식별
        v_old, rider_names, cts_name, d_orders_res, times, end_times = ProblemInput(rider_set, customer_set, env.now,
                                                                                    minus_para=True, add=add_para,
                                                                                    upper=upper, std_para=std_para,
                                                                                    std=std, mean=mean)  # 문제의 입력 값을 계산
        urgent_cts, tem1 = Basic.WhoGetPriority(un_cts, len(rider_names), env.now, time_thres=time_thres)  # C_p 계산
        expected_cts, dummy = ExpectedSCustomer(rider_set, rider_names, d_orders_res, customer_set, round(env.now,2) , toCenter = toCenter, who = 'platform')
        #peak_para에 해당하지 않는 시간대에는 보조금 문제지 않음.
        peak_para = False
        for time_slot in peak_times:
            if time_slot[0] <= env.now <= time_slot[1]:
                peak_para = True
                break
        if sorted(urgent_cts) == sorted(expected_cts) or len(urgent_cts) == 0 or len(rider_names) <= 1 or len(
                cts_name) <= 1 or No_subsidy == True:
            print('IP 풀이X', env.now)
            print('급한 고객 수:', urgent_cts, '// 예상 매칭 고객 수:', expected_cts)
            print('가능한 라이더수:', rider_names, '//고객 수:', cts_name, '//No_subsidy:', No_subsidy)
            if No_subsidy == False:
                print('가상 매칭 결과', sorted(urgent_cts), sorted(expected_cts), 'No_subsidy', No_subsidy)
            # 문제를 풀지 않아도 서비스가 필요한 고객들이 모두 서비스 받을 수 있음.
            pass
        elif peak_para == True:  # peak para 는 항상 참인 파라메터
            print('IP 풀이O', env.now)
            print('급한 고객 수:', urgent_cts, '// 예상 매칭 고객 수:', expected_cts, '//라이더 순서', d_orders_res)
            print('가능한 라이더수:', rider_names, '//고객 수:', cts_name, '//No_subsidy:', No_subsidy)
            print('V_old', np.shape(v_old), '//Time:', np.shape(times), '//EndTime:', np.shape(end_times))
            # x,v = v_info = OneClickSolver(rider_names, cts_name,v_old, riders, customers, ava_match = [],rider_seq = d_orders_res, now_time= env.now, print_gurobi = print_gurobi, priority = urgent_cts, lower = lower)
            # res, vars = subsidyASP.LinearizedSubsidyProblem(rider_names, cts_name, v_old, d_orders_res,lower_b=0, sp=urgent_cts)
            res = None
            vars = None
            # res, vars = subsidyASP.SimpleInverseSolverBasic(rider_names, cts_name, v_old, d_orders_res,times,end_times,lower_b=0, sp=urgent_cts, print_gurobi=False)
            res, vars = lpg.LinearizedSubsidyProblem(rider_names, cts_name, v_old, d_orders_res, times, end_times,
                                                     lower_b=0, sp=urgent_cts, print_gurobi=False, upper_b=upper)
            if res == False:
                # print("infeasible Try....")
                # time_con_num = list(range(0,len(rider_names)))
                time_con_num = list(range(0, len(urgent_cts)))
                time_con_num.sort(reverse=True)
                try_num = 0
                for num in time_con_num:
                    # res, vars = subsidyASP.SimpleInverseSolverBasicRelax(rider_names, cts_name, v_old, d_orders_res, times, end_times, lower_b=0, sp=urgent_cts,print_gurobi=False, time_con_num = num)
                    res, vars = lpg.LinearizedSubsidyProblem(rider_names, cts_name, v_old, d_orders_res, times,
                                                             end_times, lower_b=0, sp=urgent_cts, print_gurobi=False,
                                                             relax=num, upper_b=upper)
                    try_num += 1
                    if res != False:
                        print('Relaxing Try #', time_con_num.index(num))
                        break
                print("Try#", try_num, '//So done', len(urgent_cts) - try_num)
                # res, vars = subsidyASP.SimpleInverseSolverBasic(rider_names, cts_name, v_old, d_orders_res, times,end_times, lower_b=0, sp=urgent_cts, print_gurobi=False, time_con= False)
                # input('Check')
            if res != False:
                feasibility, res2 = solver_interpreterForIP2(res[1])
                if feasibility == True:
                    print('Fee updater')
                    FeeUpdater(res2, customer_set, rider_names, rider_set, cts_name, env.now,
                               subsidy_offer=subsidy_offer, subsidy_offer_count=subsidy_offer_count, upper=upper,
                               toCenter=toCenter)
        else:
            pass
        yield env.timeout(interval)
        # 보조금 초기화
        Basic.InitializeSubsidy(customer_set) # 보조금 초기화
        Basic.DefreezeAgent(rider_set, type = 'rider') #라이더 반영
        Basic.DefreezeAgent(customer_set, type = 'customer') #고객 반영
        if checker == False:
            print('Sys Run/T:' + str(env.now))
        else:
            input('Sys Run/T:' + str(env.now))
        #input("Check/t:" + str(env.now))