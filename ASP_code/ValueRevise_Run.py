# -*- coding: utf-8 -*-
import math
import numpy as np
import simpy
import Basic_class as Basic
import InstanceGenCL
import ValueRevise


#파라메터
speed = 2
toCenter = True
customer_wait_time = 90
fee = None
data_dir = '데이터/new_data_2_RandomCluster'
rider_intervals = InstanceGenCL.RiderGenInterval('데이터/interval_rider_data3.csv')
driver_error_pool = np.random.normal(500, 50, size=100)
customer_lamda = None
add_fee = 1500
driver_left_time = 800
driver_make_time = 200
###Running
run_time = 900


subsidy_offer = []
subsidy_offer_count = [0] * int(math.ceil(run_time / 60))  # [0]* (run_time//60) #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
env = simpy.Environment()
CUSTOMER_DICT = {}
RIDER_DICT = {}
CUSTOMER_DICT[0] = Basic.Customer(env, 0, input_location=[[36, 36], [36, 36]])
env.process(InstanceGenCL.DriverMaker(env, RIDER_DICT, CUSTOMER_DICT, end_time=run_time, speed=speed,
                                              intervals=rider_intervals[0], interval_para=True, toCenter=toCenter,
                                              run_time=driver_make_time, error=np.random.choice(driver_error_pool), pref_info = 'test_rider', driver_left_time = driver_left_time))
env.process(InstanceGenCL.CustomerGeneratorForIP(env, CUSTOMER_DICT, data_dir + '.txt', fee=fee, add_fee= add_fee))
env.process(ValueRevise.SystemRunner(env, RIDER_DICT, CUSTOMER_DICT, run_time))
env.run(until=run_time)