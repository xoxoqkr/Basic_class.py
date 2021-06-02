# -*- coding: utf-8 -*-
import Basic_class as Basic



def CustomerGeneratorForIP(env, customer_list, dir=None, end_time=1000,  customer_wait_time = 40, fee = None, lamda = None, add_fee = 0):
    datas = open(dir, 'r')
    lines = datas.readlines()
    for line in lines[3:]:
        data = line.split(',')
        store_loc = [float(data[1]), float(data[2])]
        customer_loc = [float(data[3]), float(data[4])]
        if fee == None:
            fee = int(Basic.distance(store_loc, customer_loc)/(100*120)) + 3500 #2500 #기본 수수료에 거리100m마다 150원의 추가 요금이 발생하는 요금제
        else:
            fee = fee
        fee += add_fee
        c = Basic.Customer(env, int(data[0]), input_location = [store_loc, customer_loc], fee= fee, end_time= customer_wait_time, far = int(data[6]))
        customer_list[int(data[0])] = c
        #print('Time',round(env.now,2) ,'CT#', c.name, 'gen')
        if lamda == None:
            yield env.timeout(float(data[7]))
        else:
            try:
                yield env.timeout(Basic.NextMin(lamda))
            except:
                print("lamda require type int+ or float+")
                yield env.timeout(4)
                input("check")
        if env.now > end_time:
            break



def DriverMaker(env, driver_dict, customer_set ,speed = 2, end_time = 800, intervals = [], interval_para = False, interval_res = [], toCenter = 'Back_to_center', error = 0, run_time = 900, pref_info = None, driver_left_time = 120, roulette = False):
    name = 0
    while env.now < end_time:
        rider = Basic.Rider(env, name, speed, customer_set, toCenter = toCenter, error = error, run_time = run_time, pref_info= pref_info, left_time=driver_left_time, roulette = roulette)
        driver_dict[name] = rider
        if interval_para == False:
            #print('Hr',intervals[int(env.now//60)])
            next_time = Basic.NextMin(intervals[int(env.now//60)])
            interval_res.append(next_time)
        else:
            if name >= len(intervals):
                break
            next_time = intervals[name]
        name += 1
        #print('rider', rider.name, 'gen', env.now)
        yield env.timeout(next_time)


def RiderGenInterval(dir, lamda = None):
    rider_intervals = []
    if lamda == None:
        #rider gen lamda 읽기.
        datas = open(dir, 'r')
        lines = datas.readlines()
        for line in lines[2:]:
            interval = line.split(',')
            interval = interval[2:len(interval)]
            tem = []
            for i in interval:
                # print(i)
                # input('STOP')
                tem.append(float(i))
            rider_intervals.append(tem)
    else:
        for _ in range(1000):
            rider_intervals.append(Basic.NextMin(lamda))
    return rider_intervals
