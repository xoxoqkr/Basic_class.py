# -*- coding: utf-8 -*-
import simpy
import Basic_class
import SubsidyPolicyCL
import InstanceGenCL
import ResultSaveCL
import numpy as np
import math

#필요한 인스턴스 생성
#0 시뮬레이션 환경 파라메터
run_time = 900
solver_running_interval = 10
insert_thres = 30
add_para = True
peak_times = [[0,run_time]] #결과 출력을 위한 값.
upper = 1500 #보조금 상한 지급액
checker = False
#1 라이더 관련 파라메터
speed = 1.5
wagePerHr = 9000
toCenter = None #'Back_to_center':다시 중심으로 돌아오는 시간 / 'Near_average': 근처의 평균/else: 해당 주문 만을 고려
std_para = False
#2 플랫폼 파라메터
driver_error_pool = np.random.normal(500, 50, size=100)
#3 고객 파라메터
max_food_in_bin_time = 20
customer_wait_time = 80
fee = None #Basic.distance(store_loc, customer_loc)*150 + 3500 -> 이동거리*150 + 기본료(3500)
#4 시나리오 파라메터.
Problem_states = []
ITER_NUM_list =[0]
mean_list = [0,0,0]
std_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
data_dir = '데이터/new_data_1' #'데이터/new_data_2_RandomCluster'
#datas = [[data_dir, False,'subsidy'],[data_dir, True,'normal']]
datas = [[data_dir, False,'subsidy',False],[data_dir, True,'normal',False]]
#datas = [[data_dir, False,'subsidy',False],[data_dir, True,'normal',False],[data_dir, False,'subsidy',True],[data_dir, True,'normal',True]]
# 1: 읽는 데이터 위치
# 2-3: 보조금 방식 (False, 'subsidy' : 보조금 지급/ True, 'normal' : 보조금 지급X)
# 4:roulette_para  (True : 라이더가 주문 선택에 룰렛 방식 사용 / False : 최고득점 주문 선택)

################라이더 생성##############

rider_intervals = InstanceGenCL.RiderGenInterval('데이터/interval_rider_data3.csv')
master_info = []
for i in datas:
    master_info.append([])

################실행 ##############
for data in datas:
    data_index = datas.index(data)
    print('check',data)
    for ite in ITER_NUM_list:
        #####Runiing part######
        subsidy_offer = []
        subsidy_offer_count = [0]*int(math.ceil(run_time/60)) #[0]* (run_time//60) #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        env = simpy.Environment()
        CUSTOMER_DICT = {}
        RIDER_DICT = {}
        CUSTOMER_DICT[0] = Basic_class.Customer(env, 0, input_location=[[25, 25], [25, 25]])
        env.process(InstanceGenCL.DriverMaker(env, RIDER_DICT, CUSTOMER_DICT, end_time=run_time, speed=speed, intervals= rider_intervals[0], interval_para= True,
                                              toCenter = toCenter, run_time= run_time, error= np.random.choice(driver_error_pool), roulette = data[3]))
        env.process(InstanceGenCL.CustomerGeneratorForIP(env, CUSTOMER_DICT, data[0] + '.txt', customer_wait_time=customer_wait_time, fee = fee))
        env.process(SubsidyPolicyCL.SystemRunner(env, RIDER_DICT, CUSTOMER_DICT, run_time, interval=solver_running_interval, No_subsidy = data[1], subsidy_offer=subsidy_offer,
                                                 subsidy_offer_count = subsidy_offer_count,peak_times= peak_times, upper = upper, add_para= add_para, checker= checker,
                                                 std_para = std_para, std = std_list[ite], mean = mean_list[ite], toCenter = toCenter))
        env.run(until=run_time)
        ####### 실험 종료 후 결과 저장 ########
        info = ResultSaveCL.DataSave(data, RIDER_DICT, CUSTOMER_DICT, insert_thres, speed, run_time , subsidy_offer , subsidy_offer_count , ite , mean_list , std_list, add_info= str(data[3]))
        master_info[data_index].append(info)
ResultSaveCL.DataSaver4_summary(master_info, saved_name= "res/ITE_scenario_compete_mean_"+str(mean_list[ite])+'std'+ str(std_list[ite]) + str(data[3]))
