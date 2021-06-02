# -*- coding: utf-8 -*-
import random
import simpy
import math
import operator
import numpy as np
from openpyxl import Workbook
from datetime import datetime
from ASP_code import AssignPSolver
import copy
from ASP_code import subsidyASP
from itertools import combinations
from ASP_code import ASP_class
import LinearizedASP_gurobi as lpg

class Customer(object):
    def __init__(self, env, name, input_location, end_time = 60, ready_time=2, service_time=2, fee = 1000,wait = True, far = 0, end_type = 4):
        self.name = name  # 각 고객에게 unique한 이름을 부여할 수 있어야 함. dict의 key와 같이
        self.time_info = [round(env.now, 2), None, None, None, None, end_time, ready_time, service_time]
        # [0 :발생시간, 1: 차량에 할당 시간, 2:차량에 실린 시간, 3:목적지 도착 시간,
        # 4:고객이 받은 시간, 5: 보장 배송 시간, 6:가게에서 준비시간,7: 고객에게 서비스 하는 시간]
        self.location = input_location
        self.assigned = False
        self.loaded = False
        self.done = False
        self.cancelled = False
        self.server_info = None
        self.fee = [fee, 0, None, None] # [기본 요금, 지급된 보조금, 할당된 라이더]
        self.wait = wait
        self.far = far
        self.error = 0
        self.type = random.randrange(1,end_type)
        env.process(self.Decline(env))

    def Decline(self, env, slack = 10):
        """
        고객이 생성된 후 endtime 동안 서비스를 받지 못하면 고객을 취소 시킴
        취소된 고객은 더이상 서비스 받을 수 없음.
        :param env:
        """
        yield env.timeout(self.time_info[5] + slack)
        if self.assigned == False and self.done == False:
            self.cancelled = True
            self.server_info= ["N", round(env.now,2)]

class Rider(object):
    def __init__(self, env, name, speed, customer_set, wageForHr = 9000, wait = True, toCenter = 'Back_to_center', run_time = 900, error = 0, ExpectedCustomerPreference = [1,1,1,1], pref_info = 'None', save_info = False, left_time = 120, roulette = False):
        self.env = env
        self.name = name
        self.veh = simpy.Resource(env, capacity=1)
        self.last_location = [25, 25]  # 송파 집중국
        self.served = []
        self.speed = speed
        self.wageForHr = wageForHr
        self.idle_times = [[],[]]
        self.gen_time = int(env.now)
        self.left_time = None
        self.wait = wait
        self.end_time = 0
        self.left = False
        self.earn_fee = []
        self.fee_analyze = []
        self.subsidy_analyze = []
        self.choice = []
        self.choice_info = []
        self.now_ct = 0
        self.idle_count = 0
        self.urban_point = [25,25]
        for slot_num in range(int(math.ceil(run_time / 60))):
            self.fee_analyze.append([])
            self.subsidy_analyze.append([])
        self.exp_last_location = [36,36]
        self.error = int(error)
        pref = list(range(1, len(ExpectedCustomerPreference) + 1))
        random.shuffle(pref)
        self.CustomerPreference = pref
        self.expect = ExpectedCustomerPreference
        cost_coeff = round(random.uniform(0.8,1.2),1)
        type_coeff = round(1000*random.uniform(0.8,1.2),1)
        self.coeff = [cost_coeff,type_coeff,1]
        self.p_coeff = [1,1000,1]
        env.process(self.Runner(env, customer_set, toCenter = toCenter, pref = pref_info, save_info = save_info, roulette = roulette))
        env.process(self.RiderLeft(left_time))


    def RiderLeft(self, left_time):
        """
        일을 시작한 이후 일정 시간이 되면, 기사는 시장에서 이탈.
        :param env:
        """

        yield self.env.timeout(120)
        self.left = True
        self.left_time = int(self.env.now)


    def CustomerSelector(self,customer_set, now_time, toCenter = 'Back_to_center', pref = 'None', roulette = False):
        """
        고객 중 가장 높은 가치를 가지는 고객을 선택하는 함수.
        :param customer_set: 고객 set
        :param now_time:현재 시간
        :param toCenter:다시 center로 돌아오는지 여부
        :param pref: 고객 선호를 결정
        :param roulette: 라이더의 고객 선택 방식 (1) roulette == True : roulette wheel방식 사용, (2) roulette == False : 최고 가치 선택
        :return: 가장 높은 가치를 가지는 고객 이름
        """
        ava_cts = UnloadedCustomer(customer_set, now_time)
        ava_cts_class = []
        #print('test1', ava_cts)
        ava_cts_names = []
        if len(ava_cts) > 0:
            if type(ava_cts[0]) == int:
                for ct_name in ava_cts:
                    ava_cts_class.append(customer_set[ct_name])
                ava_cts_names = ava_cts
            else:
                ava_cts_class = ava_cts
                for info in ava_cts:
                    ava_cts_names.append(info.name)
        if len(ava_cts_class) > 0:
            priority_orders = PriorityOrdering(self, ava_cts_class, now_time = self.env.now, toCenter = toCenter, who = pref)
            if roulette == True: # roulette 방식을 사용하는 경우
                value_sum = []
                roulette_wheel = []
                for ct_info in priority_orders:
                    if ct_info[1] > 0:
                        value_sum.append(ct_info[1])
                total = sum(value_sum)
                current = 0
                for ct_info in priority_orders:
                    if ct_info[1] > 0:
                        inc = ct_info[1]/total
                        roulette_wheel.append([current + inc, ct_info[0]])
                        current += inc
                rv = random.random()
                for pr_info in roulette_wheel:
                    if rv <= pr_info[0]:
                        ct = customer_set[pr_info[1]]
                        print('rv:',rv, '/val:',pr_info[0],'/sum:',total,'/len:',len(roulette_wheel))
                        #input("roulette_wheel check")
                        return ct.name, priority_orders
            else: # 최고점을 선택하는 방식을 사용하는 경우
                for ct_info in priority_orders: #이미 점수 순으로 sort 됨.
                    if ct_info[1] >= 0: #점수가 양의 값이 아니라면, 선택X
                        ct = customer_set[ct_info[0]]
                        print(self.name, 'selects', ct.name, 'at', self.env.now)
                        return ct.name, priority_orders
                    else:
                        return None, None
        return None, None


    def Runner(self, env, customer_set, wait_time=1, toCenter = 'Back_to_center', pref = 'None', save_info = False, roulette = False):
        """
        라이더가 고객을 선택하면, 고객을 서비스 하도록 수행하는 과정을 표현
        :param env:
        :param customer_set:
        :param end_time:
        :param wait_time:
        """
        while self.left == False and self.idle_count < 30:
            #print('rider test', self.name, env.now, len(customer_set),self.veh.put_queue)
            if len(self.veh.users) == 0 and self.wait == False:
                ct_name, infos = self.CustomerSelector(customer_set, env.now, toCenter = toCenter, pref = pref,  roulette = roulette)
                if ct_name != None:
                    self.choice_info.append([int(env.now), ct_name, self.last_location, infos])
                    self.idle_count = 0
                    print('Now',int(env.now),'Rider ::',self.name ,' /select::', infos)
                    #주문 수행
                    self.now_ct = ct_name
                    self.choice.append([ct_name, int(env.now)])
                    ct = customer_set[ct_name]
                    self.earn_fee.append(ct.fee[1])
                    ct.assigned = True
                    ct.time_info[1] = round(env.now, 2)
                    end_time = env.now + (distance(self.last_location, ct.location[0]) / self.speed) + ct.time_info[6]
                    end_time += ((distance(ct.location[0], ct.location[1]) / self.speed) + ct.time_info[7])
                    if int(env.now // 60) >= len(self.fee_analyze):
                        print(env.now, self.fee_analyze)
                    self.fee_analyze[int(env.now // 60)].append(ct.fee[0])
                    self.subsidy_analyze[int(env.now // 60)].append(ct.fee[1])
                    self.end_time = end_time
                    self.exp_last_location = ct.location[1]
                    #print('Rider', self.name, 'select', ct_name, 'at', env.now, 'EXP T', self.end_time)
                    #print('1:', self.last_location, '2:', ct.location)
                    #print('대기중2', self.veh.users, self.name, 'select', ct.name, 'Time:', env.now)
                    with self.veh.request() as req:
                        print('라이더',self.name,' 주문',ct.name,' 선택 /t:', int(env.now),'/대기열:',self.veh.users)
                        req.info = [ct.name, round(env.now,2)]
                        yield req  # users에 들어간 이후에 작동
                        time = distance(self.last_location, ct.location[0]) / self.speed
                        #print('With in 1:',self.last_location, '2:', ct.location[0])
                        time += ct.time_info[6]
                        end_time += time
                        ct.loaded = True
                        #ct.time_info[2] = round(env.now, 2)
                        yield env.timeout(time)
                        ct.time_info[2] = round(env.now, 2)
                        time = distance(ct.location[0], ct.location[1]) / self.speed
                        time += ct.time_info[7]
                        end_time += time
                        self.served.append([ct.name, 0])
                        #print('3:', ct.location[1])
                        yield env.timeout(time)
                        ct.time_info[3] = round(env.now, 2) - ct.time_info[7]
                        ct.time_info[4] = round(env.now,2)
                        ct.done = True
                        ct.server_info = [self.name, round(env.now,2)]
                        self.served.append([ct.name,1])
                        self.last_location = ct.location[1]
                        #임금 분석
                        print('Rider', self.name, 'done', ct_name, 'at', int(env.now))
                else: #선택할 주문이 없는 경우.
                    self.idle_count += 1
                    self.end_time = env.now + wait_time
                    self.idle_times[0].append(wait_time)  # 수행할 주문이 없는 경우
                    yield self.env.timeout(wait_time)
                    if self.idle_count > 10 and self.last_location != self.urban_point:
                        before_move = copy.deepcopy(self.last_location)
                        move_to_urban = distance(self.last_location, self.urban_point)/self.speed
                        yield env.timeout(move_to_urban) #현재 위치에서 주문이 없기 때문에, 주문이 발생할만한 곳으로 이동함.
                        self.last_location = self.urban_point
                        print(self.name, "중심가로 이동", move_to_urban,'위치',before_move,'->', self.last_location,'::i?',self.idle_count)
            else:
                self.end_time = env.now + wait_time
                self.idle_times[1].append(wait_time) #이미 수행하는 주문이 있는 경우
                yield self.env.timeout(wait_time)
        self.left = True #주문을 수행하지 못한 시간이 30분 이상인 경우


def UnloadedCustomer(customer_set, now_time):
    """
    아직 라이더에게 할당되지 않은 고객들을 반환
    :param customer_set:
    :param now_time: 현재시간
    :return: [고객 class, ...,]
    """
    res = []
    for ct_name in customer_set:
        customer = customer_set[ct_name]
        cond1 = now_time - customer.time_info[0] < customer.time_info[5]
        cond2 = customer.assigned == False and customer.loaded == False and customer.done == False
        cond3 = customer.wait == False
        #print('CT check',cond1, cond2, cond3, customer.name, customer.time_info[0])
        #input('STOP')
        if cond1 == True and cond2 == True and cond3 == True and ct_name > 0 and customer.server_info == None:
            res.append(customer)
    return res

def PriorityOrdering(veh, ava_customers, now_time = 0, minus_para = False, toCenter = 'Back_to_center', who = 'driver', last_location = None):
    """
    veh의 입장에서 ava_customers를 가치가 높은 순서대로 정렬한 값을 반환
    :param veh: class veh
    :param ava_customer_names: 삽입 가능한 고객들의 class의 list
    :return: [[고객 이름, 이윤],...] -> 이윤에 대한 내림차순으로 정렬됨.
    """
    res = []
    add_info = []
    for customer in ava_customers:
        tem = []
        #print('test',customer)
        #time = CalTime(veh.last_location, veh.speed, customer)
        if last_location == None:
            time = CalTime2(veh.last_location, veh.speed, customer, center = [25,25], toCenter = toCenter, customer_set = ava_customers)
        else:
            time = CalTime2(last_location, veh.speed, customer, center = [25,25], toCenter = toCenter, customer_set = ava_customers)
        cost = (time/60)*veh.wageForHr
        org_cost = copy.deepcopy(cost)
        fee = customer.fee[0]
        paid = 0
        t2 = time - distance(customer.location[1],[36,36])/veh.speed
        if customer.fee[2] == veh.name or customer.fee[2] == 'all':
            fee += customer.fee[1]
            paid += customer.fee[1]
        time_para = now_time + t2 < customer.time_info[0] + customer.time_info[5]
        #print("입렵 값 확인",customer.name, cost)
        #('R#',veh.name,'//CT#' ,customer.name,'//Fee$',customer.fee[0],paid, int(cost),'//Earn$',int(customer.fee[0] + paid - cost), '//ExpT',now_time + t2,'//EndT',customer.time_info[0] + customer.time_info[5], 'time_para', time_para )
        #print('check2',fee, cost, time_para)
        cost2 = 0
        if who == 'platform':
            cost2 = veh.error
        elif who == 'test_rider':
            cost2 = customer.type * veh.coeff[1]
            cost = cost*veh.coeff[0]
        elif who == 'test_platform':
            #cost2 = veh.expect[customer.type]*1000
            cost2 = customer.type * veh.p_coeff[1]
            cost = cost*veh.p_coeff[0]
        else:
            pass
        #print('고객',customer.name,'/fee', fee,'/cost:',int(cost),'/profit:', int(fee - cost - cost2),'/이동시간:',int(time))
        if minus_para == True:
            """
            if who == 'platform':
                res.append([customer.name, fee + veh.error - cost])
            else:
                res.append([customer.name, fee - cost])
            """
            res.append([customer.name, int(fee - cost - cost2), int(org_cost), int(fee)])
        elif time_para == True:
            """
            if who == 'platform':
                res.append([customer.name, fee + veh.error - cost])
            else:
                res.append([customer.name, fee - cost])
            """
            res.append([customer.name, int(fee - cost- cost2), int(org_cost), int(fee)])
        elif fee > cost + cost2 :
            res.append([customer.name, int(fee - cost - cost2), int(org_cost), int(fee)])
        else:
            #print('negative value',int(fee - cost- cost2))
            pass
    if len(res) > 0:
        res.sort(key=operator.itemgetter(1), reverse = True)
        #print(res)
    return res


def CalTime2(veh_location,veh_speed, customer, center=[25,25], toCenter = 'Back_to_center', customer_set = []):
    """
    cost(1) : customer를 서비스하는 비용
    cost(2) : 종료 후 다시 중심으로 돌아오는데 걸리는 시간.
    :param veh_location: 차량의 시작 위치
    :param veh_speed: 차량 속도
    :param customer: 고객
    :param center: 중심지의 위치(가게들이 밀접한 지역)
    :return: 필요한 시간
    """
    #print('Cal Time2',veh_location, customer.location, center)
    ready_go_t = distance(veh_location, customer.location[0]) / veh_speed
    time = ready_go_t
    order_t = distance(customer.location[0], customer.location[1]) / veh_speed
    time += order_t
    process_t = customer.time_info[6] + customer.time_info[7]
    time += process_t
    #print('현위치-가게:', int(ready_go_t), '/주문처리 이동 시간:', int(order_t), '/주문처리시간:', int(process_t))
    if toCenter == 'Back_to_center':
        time += distance(customer.location[1], center)/veh_speed
    elif toCenter == 'Near_average':
        dist = []
        for ct in customer_set:
            dist.append([ct.name, distance(customer.location[1], ct.location[0])])
        if len(dist) > 0:
            dist.sort(key=operator.itemgetter(1))
            time += dist[0][1]/veh_speed
            #aveage = []
            #for info in dist:
            #    aveage.append(info[1])
            #time += sum(aveage)/len(aveage)
    else:
        pass
    return time


def InitializeSubsidy(customer_set):
    """
    customer_set의 보조금을 모두 초기화 한다.
    :param customer_set:
    :return:
    """
    for ct_name in customer_set:
        ct = customer_set[ct_name]
        if ct.assigned == False or ct.time_info == None:
            ct.fee[1] = 0
            ct.fee[2] = None
    return None

def DefreezeAgent(object_dict, type = 'rider'):
    """
    object_dict에 있는 고객들을 subsidy문제가 고려할 수 있는 상태로 변환
    :param object_dict:
    :param type:
    :return:
    """
    if type == 'rider':
        for object_name in object_dict:  # interval이 끝난 후 새롭게 들어온 라이더와 고객을 시스템에 반영시킴
            rider = object_dict[object_name]
            if rider.left == False and rider.wait == True:
                rider.wait = False
    elif type == 'customer': #type == 'customer'
        for object_name in object_dict:
            customer = object_dict[object_name]
            if customer.wait == True and customer.name > 0:
                customer.wait = False
    else:
        input("Agent name error")
    return None

def AvaRider(rider_set, now_time,interval = 10):
    res = []
    for rider_name in rider_set:
        rider = rider_set[rider_name]
        cond1 = rider.wait == False
        cond2 = now_time < rider.end_time < now_time + interval
        cond3 = rider.left == False
        #print(rider.name, '::', cond1, '::',cond2, '::',cond3)
        if cond1 == True and cond2 == True and cond3 == True:
            res.append(rider.name)
            #print(rider.name,'::',rider.end_time ,'<', now_time + interval)
    print('AvaRider#', len(res))
    return res


def WhoGetPriority(cts_names , ava_riders_num, now_time, time_thres = 0.8):
    """
    현재 TW가 임박한 고객을 계산
    :param cts_names: [고객 class,...,]
    :param customer_set:
    :param ava_riders_num:가능한 라이더의 수(이 수만큼만 고객들을 뽑기 때문)
    :param now_time: 현재 시간
    :param time_thres: 고객의 endtime 중 얼마의 시간이 지난 경우 우선 고객으로 지정하는가? (0<x<1)
    :return: [고객 이름1, 고객 이름2,...ㅡ]
    """
    scores = []
    index = 0
    print('candidates',len(cts_names),'ava_riders_num', ava_riders_num)
    test = []
    for customer in cts_names:
        test.append(int(now_time - customer.time_info[0]))
        required_time = distance(customer.location[0], customer.location[1])/1.5 + customer.time_info[6] + customer.time_info[7]
        if (customer.time_info[0] + customer.time_info[5] - now_time)*time_thres > required_time :
        #if now_time - customer.time_info[0] > customer.time_info[5]*time_thres:
            scores.append([customer.name, customer.time_info[0], index])
        index += 1
    print('urgent ct info',scores)
    #print('t test', test)
    if len(scores) > 0:
        scores.sort(key=operator.itemgetter(1))
        res = []
        tem = []
        for info in scores[:ava_riders_num]:
            res.append(info[2])
            tem.append(info[0])
        return res, tem
    return [], []


def distance(x1, x2):
    return round(math.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2), 2)


def NextMin(lamda):
    """
    :rtype: object
    """
    # lambda should be float type.
    # lambda = input rate per unit time
    next_min = (-math.log(1.0 - random.random()) / lamda)*60
    return float("%.2f" % next_min)

