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
import simpy
import Basic_class as Basic
import SubsidyPolicyCL
import InstanceGenCL
import ResultSaveCL


def InputCalculate(expectation, actual_result):
    R = []
    E = []
    #플랫폼이 예상한 선택 정보
    for info in expectation:
        if info == 1:
            E.append(1)
        else:
            E.append(0)
    #라이더가 실제 선택한 정보
    for info in actual_result:
        if info == 1:
            R.append(1)
        else:
            R.append(0)
    return R, E
#[int(env.now), ct_name, self.last_location , rev_infos]
# rev_infos = [고객 이름, fee-cost - cost2, cost, fee]

def InputCalculate2(rider, customer_set, index = -1):
    """
    select_coefficient = []
    select_info = rider.choice_info[3][0]
    dist_cost = select_info[2]
    customer_type = customer_set[select_info[0]].type
    type_cost = rider.expect[customer_type] * 100
    fee = select_info[3]
    select_coefficient = [dist_cost, type_cost, fee]
    #다른 고객들에 대하여
    other_coefficient = []
    """
    rev_infos = []
    print('Index#',index,'/rider_choice: ',rider.choice_info[index])
    for info in rider.choice_info[index][3]:
        ct_name = info[0]
        dist_cost = info[2]
        customer_type = customer_set[ct_name].type
        type_cost = rider.expect[customer_type]
        fee = info[3]
        rev_infos.append([-dist_cost, -type_cost, fee])
    return rev_infos[0], rev_infos[1:]

def ChoiceCheck(rider, customer_set):
    """
    라이더가 고객 선택을 제대로 한 것인지 확인.
    :param rider:
    :return:
    """
    #1 자신이 선택할 수 있었던 것 중 최고를 선택하는가?
    print('rider',rider.name,'.choice_info',rider.choice_info)
    for raw_info in rider.choice_info:
        info = raw_info[3]
        print('info',info)
        select = info[0]
        others = info[1:]
        #select_score =
        #print('select',select)
        #print('others',others)
        other_score = []
        for other in [select]+others:
            coeff_index = 2
            value = 0
            value += -other[2]* rider.p_coeff[0]
            value += -customer_set[other[0]].type * rider.p_coeff[1]
            value += other[3] * rider.p_coeff[2]
            """
            for coeff in rider.p_coeff[:2]:
                value += other[coeff_index]*coeff
                coeff_index += 1
            """
            other_score.append([other[0],round(value,1)]) #[[고객 이름, 이윤],...]
        #print('P예상 선택함',other_score[0])
        #other_score = other_score[1:]
        other_score.sort(key=operator.itemgetter(1))
        other_score.reverse()
        print('P예상 선택함',other_score[0],'P예상 후보들', other_score)
        #실제 계산 부.
        actual_score = []
        for other in [select]+others:
            coeff_index = 2
            value = 0
            value += -other[2]* rider.coeff[0]
            value += -customer_set[other[0]].type * rider.coeff[1]
            value += other[3] * rider.coeff[2]
            """
            for coeff in rider.p_coeff[:2]:
                value += other[coeff_index]*coeff
                coeff_index += 1
            """
            actual_score.append([other[0],round(value,1)])
        #print('R예상 선택함',actual_score)
        #actual_score = actual_score[1:]
        actual_score.sort(key=operator.itemgetter(1))
        actual_score.reverse()
        print('R예상 후보들', actual_score)


    return None


def Comapre2List(base, compare):
    #base에는 있으나, compare는 없는 요소 찾기.
    expectation = []
    actual = []
    for info1 in base:
        for info2 in compare:
            if info1[0] == info2[0]:
                if info1[1] == info2[1]:
                    pass
                else:
                    expectation.append(info1)
                    actual.append(info2)
                break
    return expectation, actual


def RiderChoiceWithInterval(rider_set, rider_names, now, interval = 10):
    """
    [now - interval, now)까지의 라이더가 선택한 고객을 반환
    [[라이더이름, 고객이름],...]
    :param rider_set:
    :param rider_names:
    :param now:
    :param interval:
    :return:
    """
    #print(now - interval,'->',now, "라이더", rider_names)
    actual_choice = []
    for rider_name in rider_names:
        rider = rider_set[rider_name]
        #print('rider.choice', rider.choice)
        for info in rider.choice:
            #print(info[1])
            if now - interval < info[1] <= now:
                #print(info)
                actual_choice.append([rider.name, info[0]])
                print("차이 발생",[rider.name, info[0]])
                break
            else:
                pass
    return actual_choice


def SystemRunner(env, rider_set, customer_set, cool_time, interval=10, checker = False, toCenter = 'Back_to_center'):
    # 보조금 부문을 제외하고 단순하게 작동하는 것으로 시험해 볼 것.
    while env.now <= cool_time:
        #플랫폼이 예상한 라이더-고객 선택
        un_served_ct = Basic.UnloadedCustomer(customer_set, env.now)
        ava_rider = Basic.AvaRider(rider_set, env.now)
        print("시간::",env.now," / 서비스가 필요한 고객 수::", len(un_served_ct), " / interval동안 빈 차량 수::", len(ava_rider))
        v_old, rider_names, cts_name, d_orders_res, times, end_times = SubsidyPolicyCL.ProblemInput(rider_set, customer_set, env.now,minus_para=True)  # 문제의 입력 값을 계산
        print("선택순서", d_orders_res)
        expected_cts, platform_expected = SubsidyPolicyCL.ExpectedSCustomer(rider_set, rider_names, d_orders_res, customer_set, round(env.now, 2), toCenter=toCenter,
                          who='test_platform')
        print('예상 고객 점검',expected_cts, platform_expected)
        #platform_expected = [[rider_name, 고객 이름],...,]
        yield env.timeout(interval)
        now = round(env.now, 1)
        #선택된 라이더들에 대한 가치 함수 갱신
        actual_choice = RiderChoiceWithInterval(rider_set, rider_names, now, interval=interval)
        expectation, actual = Comapre2List(platform_expected, actual_choice)
        print("차이발생/T now:",now,"/예상한 선택", expectation , "/실제 선택", actual)
        if len(expectation) > 0:
            input("계산 확인.")
            for info in expectation:
                rider = rider_set[info[0]]
                selected, others = InputCalculate2(rider, customer_set)
                ChoiceCheck(rider, customer_set)
                past_choices = []
                indexs = list(range(len(rider.choice_info) - 1))
                indexs.reverse()
                for index1 in indexs:
                    past_select, past_others = InputCalculate2(rider, customer_set, index=index1)
                    if len(past_others) > 0:
                        past_choices.append(
                            [past_select, past_others])  # 비교 대상이 필요함. 만약 past_others = empty 라면, 비교 식을 선언할 수 없음.
                    print("추가되는 사례", len(past_choices))
                    print('past_choices', past_choices)
                    input("추가 사례 확인")
                input("계산 확인2.")
                feasibility, res = lpg.ReviseCoeffAP(selected, others, rider.p_coeff, past_data=past_choices)
                # 계수 갱신
                if feasibility == True:
                    for index in range(len(res)):
                        rider.p_coeff[index] += res[index]
                    print("갱신됨::", res)
            pass
        """
        #라이더별로 revise문제 정의 및 풀기
        print("현재 시점", int(env.now), '::', len(platform_expected),len(actual_choice))
        for index in range(min(len(platform_expected),len(actual_choice))):
            rider = rider_set[rider_names[index]]
            print("예상 고객", platform_expected, "Vs 실제 고객", actual_choice,': 수행 고객수', len(rider.choice_info))
            #print("예상 고객",platform_expected[index][1],"Vs 실제 고객",actual_choice[index][1])

            if platform_expected[index] in actual_choice:
            #if platform_expected[index][1] == actual_choice[index][1]:
                print("예상과 선택이 일치", platform_expected[index])
                pass
            elif len(rider.choice_info) > 0:
                print("계산 필요", platform_expected[index])
                #예상과 실제가 다름 -> 문제 계산 필요
                # 문제 입력값 계산
                selected, others = InputCalculate2(rider, customer_set)
                ChoiceCheck(rider, customer_set)
                input("계산 확인")
                # 문제 풀이
                #past_choices_infos = rider.choice[:len(rider.choice) - 1]
                past_choices = []
                indexs = list(range(len(rider.choice_info) - 1))
                indexs.reverse()
                print('사용사능한 indexs', indexs, 'Len:', len(rider.choice_info))
                print('플랫폼 계수',rider.p_coeff)
                print('라이더 계수', rider.coeff)
                print(selected, others)
                for index1 in indexs:
                    past_select, past_others = InputCalculate2(rider, customer_set, index = index1)
                    if len(past_others) > 0:
                        past_choices.append([past_select, past_others]) #비교 대상이 필요함. 만약 past_others = empty 라면, 비교 식을 선언할 수 없음.
                    print("추가되는 사례",len(past_choices))
                    print('past_choices',past_choices)
                    input("추가 사례 확인")
                feasibility , res = lpg.ReviseCoeffAP(selected, others, rider.p_coeff, past_data = past_choices)
                #계수 갱신
                if feasibility == True:
                    for index in range(len(res)):
                        rider.p_coeff[index] += res[index]
                    print("갱신됨::", res)
            else:
                pass
        """
        # 보조금 초기화
        Basic.InitializeSubsidy(customer_set)  # 보조금 초기화
        Basic.DefreezeAgent(rider_set, type='rider')  # 라이더 반영
        Basic.DefreezeAgent(customer_set, type='customer')  # 고객 반영
        if checker == False:
            print('Sys Run/T:' + str(env.now))
        else:
            input('Sys Run/T:' + str(env.now))
        #input("정보확인1")