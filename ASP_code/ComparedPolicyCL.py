# -*- coding: utf-8 -*-
import random
import Basic_class as Basic


def AllSubsidy(customer_set: dict, now_time: int,
               subsidy_info: list = [[0.4, 0.3, 0.1], [0.3, 0.2, 0.2], [0.2, 0.1, 0.3], [0.1, 0, 0.4]],
               subsidy_offer: list = [],
               subsidy_offer_count: list = []) -> object:
    """
    subsidy_info에 따라 서비스 받지 못한 고객 모두에게 보조금을 지급
    subsidy_info는 [보조금 지급 시작 구간, 보조금 지급 종료 구간, fee대비 보조금%]이다.
    :param customer_set: 고객 집합
    :param now_time: 현재시간
    :param subsidy_info: 보조금 정보
    :param subsidy_offer: 제안된 보조금
    :param subsidy_offer_count: 제안된 보조금 수
    """
    customers = Basic.UnloadedCustomer(customer_set, now_time)
    index, ct_names = Basic.WhoGetPriority(customers, 100, now_time)
    for ct_name in ct_names:
        ct = customer_set[ct_name]
        remain_time = (ct.time_info[0] + ct.time_info[5]) - now_time
        ratio = remain_time / ct.time_info[5]  # 작아질 수록 촉박함
        for info in subsidy_info:
            if info[1] <= ratio < info[0]:
                subsidy = ct.fee[0] * info[2]
                ct.fee[1] = subsidy
                ct.fee[2] = 'all'
                subsidy_offer.append(['all', subsidy])
                subsidy_offer_count[int(now_time // 60)] += 1
                break
    return subsidy_offer, subsidy_offer_count

def RandomSubsidy(customer_set: dict, now_time: int, subsidy_info: object = [0.3, 0], max_ratio: int = 1, subsidy_offer: list = [],
                  subsidy_offer_count: list = []) -> object:
    """
    고객들에 대하여, 고객의 남은 시간이 subsidy_info인 경우 fee*(0.3~max_ratio)갑의 보조금을 제안
    :param customer_set: 고객 집합
    :param now_time: 현재 시간
    :param subsidy_info: 보조금 정보
    :param max_ratio: 보조금 상한
    :param subsidy_offer: 제안된 보조금
    :param subsidy_offer_count: 제안된 보조금 수
    """
    customers = Basic.UnloadedCustomer(customer_set, now_time)
    index, ct_names = Basic.WhoGetPriority(customers, 100, now_time)
    for ct_name in ct_names:
        ct = customer_set[ct_name]
        remain_time = (ct.time_info[0] + ct.time_info[5]) - now_time
        ratio = remain_time / ct.time_info[5]  # 작아질 수록 촉박함
        if subsidy_info[1] <= ratio < subsidy_info[0]:
            subsidy = random.randrange(int(ct.fee[0] * 0.3), int(ct.fee[0] * max_ratio))
            ct.fee[1] = subsidy
            ct.fee[2] = 'all'
            subsidy_offer.append(['all', subsidy])
            subsidy_offer_count[int(now_time // 60)] += 1
    return subsidy_offer, subsidy_offer_count

def SystemRunner(env, rider_set, customer_set, cool_time, interval=10, subsidy_type='all', subsidy_offer=[],
                      subsidy_offer_count=[]):
    """
    보조금 지급 방식 ['all', 'random'] 중 한 가지 방식을 사용하는 배송 플랫폼을 시뮬레이션 하는 기능.
    :param env: simpy의 환경
    :param rider_set: 라이더들
    :param customer_set: 고객들(dict)
    :param cool_time: 시뮬레이션 쿨 타임. 이 시간 이후에는 보조금 지급X
    :param interval: 보조금을 제안하는 의사결정이 이루어지는 시간 간격
    :param subsidy_type:
    :param subsidy_offer: 제안된 보조금
    :param subsidy_offer_count: 제안된 보조금 수
    """
    while env.now < cool_time:
        if subsidy_type == 'all':
            subsidy_offer, subsidy_offer_count = AllSubsidy(customer_set, env.now, subsidy_offer=subsidy_offer, subsidy_offer_count=subsidy_offer_count)
        elif subsidy_type == 'random':
            subsidy_offer, subsidy_offer_count = RandomSubsidy(customer_set, env.now, subsidy_offer=subsidy_offer,subsidy_offer_count=subsidy_offer_count)
        else:
            input('error')
        yield env.timeout(interval)
        Basic.InitializeSubsidy(customer_set) # 보조금 초기화
        Basic.DefreezeAgent(rider_set, type = 'rider') #라이더 반영
        Basic.DefreezeAgent(customer_set, type = 'customer') #고객 반영


