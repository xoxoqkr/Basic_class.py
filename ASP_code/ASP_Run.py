# -*- coding: utf-8 -*-
from ASP_code import ASP_class
import simpy
import  random
#필요한 인스턴스 생성
speed = 1.5
run_time = 800
customer_wait_time = 60
solver_running_interval = 10
max_food_in_bin_time = 20
insert_thres = 30
max_on_hand_order = 3
wagePerHr = 10000
Problem_states = []
minimum_add_fee = 800
ITER_NUM = 10
C_p_para = 2 # 1: 가장 효과적인 고객을 할당, 2: 임박한 고객을 할당.
end_time = customer_wait_time + insert_thres
all_subsidy_info = [[end_time/3, (end_time*1.5)/3, 0.05], [(end_time*1.5)/3, (end_time*2)/3, 0.1], [(end_time*2.5)/3, end_time, 0.15]]

"""
for i in range(1,10):
    coord = []
    for j in range(2):
        tem = []
        for k in range(2):
            tem.append(random.randrange(0,50))
        coord.append(tem)
    CUSTOMER_DICT[i] = ASP_class.Customer(env, i, input_location = coord, end_time = run_time)
    if i % 10 == 0:
        print(i, 'th customer gen', env.now)

env.timeout(10)

for i in range(2):
    RIDER_DICT[i] = ASP_class.Driver(env, i, speed, CUSTOMER_DICT)

env.timeout(10)
env.run(until= run_time)
"""
#store_pr = [0.5, 0.5]
exp_days = 1
#data_set = [['ASP_instance/IP_C_600', True,1,2]]
data_set = []
#data_set += [['ASP_instance/IP_C_600', True,1,4],['ASP_instance/IP_R_600', True,1,4],['ASP_instance/IP_RC_600', True,1,4]]
#data_set += [['ASP_instance/IP_C_600', True,1,1],['ASP_instance/IP_C_600', True,1,2],['ASP_instance/IP_C_600', True,1,3],['ASP_instance/IP_C_600', True,1,4]]
#data_set += [['ASP_instance/IP_R_600', True,1,1],['ASP_instance/IP_R_600', True,1,2],['ASP_instance/IP_R_600', True,1,3],['ASP_instance/IP_R_600', True,1,4]]
data_set += [['ASP_instance/IP_RC_600', True,1,1,0.3],['ASP_instance/IP_RC_600', True,1,1,0.4],['ASP_instance/IP_RC_600', True,1,1,0.5],['ASP_instance/IP_RC_600', True,1,1,0.6],['ASP_instance/IP_RC_600', True,1,1,0.7],['ASP_instance/IP_RC_600', True,1,1,0.8],['ASP_instance/IP_RC_600', True,1,1,0.9]]
data_set += [['ASP_instance/IP_RC_600', True,1,1,0.7],['ASP_instance/IP_RC_600', True,1,1,0.8],['ASP_instance/IP_RC_600', True,1,2],['ASP_instance/IP_RC_600', True,1,3],['ASP_instance/IP_RC_600', True,1,4]]
#data_set += [['ASP_instance/IP_C_600', True,1,1],['ASP_instance/IP_R_600', True,1,1],['ASP_instance/IP_RC_600', True,1,1]]
#data_set = [['ASP_instance/IP_C_600', True,1],['ASP_instance/IP_C_600', False,1],['ASP_instance/IP_C_600', True,2],['ASP_instance/IP_C_600', False,2],['ASP_instance/IP_C_600', True,3], ['ASP_instance/IP_C_600', False,3]] #['ASP_instance/IP_C_600', True,2]
#data_set = [['ASP_instance/IP_C_600', True,1],['ASP_instance/IP_C_600', False,1],['ASP_instance/IP_C_600', True,2],['ASP_instance/IP_C_600', False,2],['ASP_instance/IP_C_600', True,3],['ASP_instance/IP_C_600', False,3]]#['IP_RC_600','IP_R_600','IP_C_600']
#para2: True :원 모형 풀이 / False = 변형 모형 풀이
##revise_rule: 어떤 방식으로 갱신하는가 1:평균 리드타임에 반비례 2:평균 서비스한 고객 수에 비례 3:성공률(각자 플랫폼이 받은 고객 수 대비 성공한 고객의 비율)
interval_eg = [23, 28, 21, 20, 10, 27, 18, 11, 13, 15, 11, 24, 11, 27, 22, 11, 26, 28, 18, 10, 27, 16, 24, 24, 10, 19, 13, 28, 21, 25, 12, 20, 20, 12, 28, 15, 13, 13, 28, 23, 14, 23]
interval_lamda_base =[2,2,5,4,7,4,3,2,2,4,7,7,3,3]
#interval_lamda = []
interval_muls = [0.6,0.8,1,1.2,1.4,1.6,1.8,2]
for interval_mul in interval_muls[3:7]:
    interval_lamda = []
    for i in interval_lamda_base:
        interval_lamda.append(i*interval_mul)
    ITE_INFO = []
    for data in data_set:
        ITE_INFO.append([])
    for ite in range(0, ITER_NUM):
        sc_infos = []
        intervals = []
        #intervals = interval_eg
        for data in data_set:
            day_data = []
            store_pr = [0, 1] #이 값이 1,0이면 1개의 플랫폼만 존재 [보조금X플랫폼의 선택 확률, 보조금을 주는 플랫폼의 선택확률]
            solver_type = data[1]
            revise_para = data[2]
            for day in range(exp_days):
                env = simpy.Environment()
                CUSTOMER_DICT = {}
                RIDER_DICT = {}
                CUSTOMER_DICT[0] = ASP_class.Customer(env, 0, input_location=[[25, 25], [25, 25]])
                #env.process(ASP_class.DriverMaker(env, RIDER_DICT, CUSTOMER_DICT, end_time = run_time, speed = speed, thres = insert_thres, max_on_hand_order = max_on_hand_order))
                #print('interval',data_set.index(data), intervals)
                #print('interval',data_set.index(data)%4, intervals)
                #input('stop')
                if data_set.index(data)%10 == 0 and intervals == []:
                    env.process(ASP_class.DriverMaker(env, RIDER_DICT, CUSTOMER_DICT, end_time=run_time, speed=speed,
                                                      thres=insert_thres, max_on_hand_order=max_on_hand_order, intervals = interval_lamda, interval_res= intervals))
                else:
                    env.process(ASP_class.DriverMaker(env, RIDER_DICT, CUSTOMER_DICT, end_time=run_time, speed=speed,
                                                      thres=insert_thres, max_on_hand_order=max_on_hand_order, intervals = intervals, interval_para= True))
                #env.process(ASP_class.CustomerMaker(env,CUSTOMER_DICT,300,50, end_time = run_time, speed = speed, dist_thres = max_food_in_bin_time))
                env.process(ASP_class.CustomerGeneratorForIP(env, CUSTOMER_DICT, data[0] + '.txt', speed= speed, wagePerHr= wagePerHr, select_pr = store_pr, customer_wait_time=customer_wait_time))
                #print("NOW", data[3])
                #input("STOP")
                if data[3] == 1:
                    #기존의 방식
                    C_p_time_thres = data[4]
                    env.process(ASP_class.PlatformRunner(env,RIDER_DICT,CUSTOMER_DICT,end_time = run_time, run_interval = solver_running_interval, Problem_states = Problem_states, lower = minimum_add_fee, para2= solver_type, thres = insert_thres, C_p_para= C_p_para,
                                                         C_p_time_thres = C_p_time_thres))
                elif data[3] == 2:
                    #random_subsidy
                    env.process(ASP_class.RandomSubsidy(env, RIDER_DICT, CUSTOMER_DICT, cp_t=30, max_s=0.15, run_interval=10, end_time = run_time))
                elif data[3] == 3:
                    #all_subsidy
                    env.process(ASP_class.AllSubsidy(env, RIDER_DICT, CUSTOMER_DICT, inc_ratio_step = all_subsidy_info,run_interval=10, end_time = run_time))
                else:
                    print("No subsidy")

                env.run(until= run_time)
                ASP_class.DataSaveAsXlxs(RIDER_DICT, CUSTOMER_DICT,Problem_states,veh_write = True, ct_write = True, add_infos = data[0] + ';MaxOnHand;'+str(data[3])+';')
                if 1 not in store_pr:
                    store_pr, day_info = ASP_class.DayByDayStore(CUSTOMER_DICT, store_pr, 0.15, day, time_thres= 200, revise_rule = revise_para)
                    day_data += day_info
                print('day done',store_pr)
                #input('check next pr')
                this_sc_info = ASP_class.DataSaver2(data, CUSTOMER_DICT, RIDER_DICT,insert_thres, speed, now_time= int(env.now))
                sc_infos.append(this_sc_info)
            print('days_test_'+ str(solver_type) + '_' + str(revise_para))
            #input('CHECK')
            ASP_class.daysDataSave(day_data, add_infos='days_test_'+ str(solver_type) + '_' + str(revise_para))
            ASP_class.DataSaver2_summary(sc_infos)
            #print(data)
            #print(sc_infos)
            #input('stop')
            save_index = data_set.index(data)
            sc_infos[-1][0] += str(interval_mul)
            ITE_INFO[save_index].append(sc_infos[-1])
            #ITE_INFO[save_index].append(interval_mul)
            print('intervals',intervals)
            #input('stop')
    ASP_class.DataSaver3_summary(ITE_INFO)