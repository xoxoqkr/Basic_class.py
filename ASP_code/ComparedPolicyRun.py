# -*- coding: utf-8 -*-
import simpy
import Basic_class
import ComparedPolicyCL


#필요한 인스턴스 생성
speed = 1.5
run_time = 900
customer_wait_time = 80
solver_running_interval = 10
max_food_in_bin_time = 20
insert_thres = 30
max_on_hand_order = 3
wagePerHr = 9000
Problem_states = []
minimum_add_fee = 900
fee = None
ITER_NUM = 3
#ITER_NUM_list = range(0,ITER_NUM)
ITER_NUM_list =[0,1]
add_para = True
toCenter = False
#No_subsidy = True
end_time = customer_wait_time + insert_thres
all_subsidy_info = [[end_time/3, (end_time*1.5)/3, 0.05], [(end_time*1.5)/3, (end_time*2)/3, 0.1], [(end_time*2.5)/3, end_time, 0.15]]
peak_times = [[0,run_time]]#[[180,300],[660,800]]
#mean_list = [-0.3,-0.2,-0.1,0,0.1,0.2,0.3]
mean_list = [0,0,0]
#std = 1
std_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
#std_list = [0.2,0.5,1.0,1.5,2.0]
std_para = False
data_dir = '데이터/new_data_2_RandomCluster'
#datas = [['ASP_instance/IP_RC_600', False],['ASP_instance/IP_RC_600', True]]
#datas = [['데이터/new_data_1', False,'subsidy'],['데이터/new_data_1', True,'normal'],['데이터/new_data_1', True,'all'],['데이터/new_data_1', True,'random']]
#datas = [['데이터/new_data_1', True,'normal']]
#datas = [['데이터/new_data_2_Random', False,'subsidy']]
#datas = [['데이터/new_data_2_Random', True,'normal'],['데이터/new_data_2_Random', True,'all'],['데이터/new_data_2_Random', True,'random']]
#datas = [['데이터/new_data_2_Random', False,'subsidy']]
datas = [[data_dir, True,'all'],[data_dir, True,'random']]
#datas = [['데이터/new_data_1', True,'normal'],['데이터/new_data_1', True,'all'],['데이터/new_data_1', True,'random']]
#datas = [['데이터/new_data_1', True,'normal'],['데이터/new_data_1', True,'all'],['데이터/new_data_1', True,'random']]
upper = 1500
checker = False
#datas = [['데이터/new_data_1', True,'all'],['데이터/new_data_1', True,'random']]
#datas = [['데이터/new_data_1', False],['데이터/new_data_1', False]]

#Interval 을 저장 된 csv에서 읽어 오기

rider_intervals = Basic_class.RiderGenInterval('데이터/interval_rider_data3.csv')
master_info = []
for i in datas:
    master_info.append([])
for data in datas:
    data_index = datas.index(data)
    print('check',data)
    #input('STOP')
    for ite in ITER_NUM_list:
        #####Runiing part######
        subsidy_offer = []
        subsidy_offer_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        env = simpy.Environment()
        CUSTOMER_DICT = {}
        RIDER_DICT = {}
        CUSTOMER_DICT[0] = Basic_class.Customer(env, 0, input_location=[[36, 36], [36, 36]])
        env.process(Basic_class.DriverMaker(env, RIDER_DICT, CUSTOMER_DICT, end_time=run_time, speed=speed, intervals= rider_intervals[0], interval_para= True, toCenter = toCenter))
        env.process(Basic_class.CustomerGeneratorForIP(env, CUSTOMER_DICT, data[0] + '.txt', customer_wait_time=customer_wait_time, fee = fee))


        if data[2] in ['subsidy', 'normal']:
            env.process(Basic_class.SystemRunner(env, RIDER_DICT, CUSTOMER_DICT, run_time, interval=solver_running_interval, No_subsidy = data[1], subsidy_offer=subsidy_offer, subsidy_offer_count = subsidy_offer_count,peak_times= peak_times, upper = upper, add_para= add_para, checker= checker, std_para = std_para, std = std_list[ite], mean = mean_list[ite], toCenter = toCenter))
        elif data[2] in ['all', 'random']:
            env.process(Basic_class.SystemRunner2(env, RIDER_DICT, CUSTOMER_DICT, run_time, interval=solver_running_interval, subsidy_type= data[2],subsidy_offer=subsidy_offer,subsidy_offer_count=subsidy_offer_count))
        else:
            pass
        #env.process(Basic_class.SystemRunner2(env, RIDER_DICT, CUSTOMER_DICT, 800))
        env.run(until=run_time)
        fees = []
        subsidy_take_num = []
        for ct_name in CUSTOMER_DICT:
            customer = CUSTOMER_DICT[ct_name]
            fees.append(customer.fee[0])
            if customer.server_info != None and customer.server_info[0] == customer.fee[2]:
                subsidy_take_num.append([customer.name,customer.server_info[0], customer.fee[1]])
        print('Ave fee', sum(fees)/len(fees))
        print('subsidy_take_num', len(subsidy_take_num))
        info = Basic_class.SingleDataSaver(data, CUSTOMER_DICT, RIDER_DICT ,insert_thres, speed, run_time)
        info.append(len(subsidy_offer))
        info += subsidy_offer_count
        master_info[data_index].append(info)
        #Basic_class.DataSaver4_summary([info])
        f = open(data[2]+str(ite) +"mean_"+str(mean_list[ite])+"std_"+str(std_list[ite])+ "_ite_info_save.txt",'a')
        for ele in info:
            f.write(str(ele)+'\n')
        f.close
        Basic_class.CustomerDataSaver(CUSTOMER_DICT, data[2]+str(ite)+"mean_"+str(mean_list[ite])+"std_"+str(std_list[ite])+ "_ct_save")
Basic_class.DataSaver4_summary(master_info, saved_name= "mean_"+str(mean_list[ite])+'std'+ str(std_list[ite]))