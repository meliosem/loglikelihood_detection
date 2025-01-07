import numpy as np
import pandas as pd
import scipy
from scipy import stats
import scipy.special
from scipy.special import comb
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.tsatools import detrend
from statsmodels.tsa.seasonal import STL
from numpy import linalg as LA
import numpy.fft as nf
import pymannkendall as mk
import sys
sys.path.append("./mechanism/")
import grr, oue, compromised_sr, compromised_pm, sw_compromised
import OLH_compromised_user_matrix_implemented as OLH_compromised_user
import OLH_compromised_server_matrix_implemented as OLH_compromised_server
from sklearn import datasets, linear_model
import time


def norm_sub(n, aggregated_result):  # this is count, not frequency
    # print("norm_sub")
    # result = np.sum(noisy_samples, axis = 0)
    # result = (result - n*q) / (p-q)
    estimates = np.copy(aggregated_result)
    while (np.fabs(sum(estimates) - n) > 1) or (estimates < 0).any(): # Norm-Sub
        estimates[estimates < 0] = 0
        total = sum(estimates)
        # print(total)
        mask = estimates > 0
        # print("sum(mask)", sum(mask))
        diff = (n - total) / sum(mask)
        estimates[mask] += diff

    return estimates, n


def mle_apx(eps, n, est_dist):
    p = 1 / 2
    q = 1 / (np.exp(eps) + 1)
    estimates = np.copy(est_dist) / n

    while (np.fabs(sum(estimates * n) - n) > 1) or (estimates < 0).any():
        estimates[estimates < 0] = 0
        pos_f_index = estimates > 0
        pos_f_sum = np.sum(pos_f_index)
        pos_f = estimates[pos_f_index]
        x = (np.sum(pos_f) * (p - q) - (p - q)) / ((p - q) * (1 - p - q) - pos_f_sum * q * (1 - q))
        gamma = (p - q) / (p - q + (p - q) * (1 - p - q) * x)
        delta = (q * (1 - q) * x) / (p - q + (p - q) * (1 - p - q) * x)

        mask = estimates > 0
        estimates[mask] = estimates[mask] * gamma + delta
    return estimates * n



def synthesize_discrete_data(n, distribution, domain):
    # data = np.array([])
    # for i in range(len(distribution)):
    #     num = int(n * distribution[i]) + 1
    #     temp = np.array([int(i)] * num)
    #     data = np.append(data, temp)
    # data = data.astype("int")

    data = np.random.choice(np.arange(domain), p=distribution, size=n)
    data = data.astype("int")
    np.random.shuffle(data)
    return data

def synthesize_numerical_data(size, dist):
    num_bin = len(dist)
    bin_width = 1/num_bin
    sampled_data = [np.random.uniform(i * bin_width, (i+1) * bin_width, int(size * dist[i])+1) for i in range(num_bin)]
    sampled_data = np.array(np.concatenate(sampled_data).flat)
    if np.size(sampled_data) > size:
        np.random.shuffle(sampled_data)
        sampled_data = sampled_data[:size - np.size(sampled_data)]
    return sampled_data




def EM(n, ns_hist, transform, max_iteration, loglikelihood_threshold, init_dist):
    theta = np.ones(n) / float(n)

    if np.size(init_dist) != 0:
        return init_dist
        # theta = 0.6*init_dist + 0.4*np.ones(n) / float(n)
    # print("theta", theta)

    theta_old = np.zeros(n)
    r = 0
    sample_size = sum(ns_hist)
    # print("sample size", sample_size)
    old_logliklihood = 0

    while (LA.norm(theta_old - theta, ord=1) > 1 / sample_size) and (r < max_iteration):
    # while (LA.norm(theta_old - theta, ord=1) > 1 / sample_size) or (r < max_iteration):
    # while r < max_iteration:
    #     print("theta", theta)
        begin_time = time.time()
        theta_old = np.copy(theta)
        X_condition = np.matmul(transform, theta_old)
        # print("X_condition:", X_condition)
        TMP = transform.T / X_condition
        # print("TMP:", TMP[3])

        P = np.copy(np.matmul(TMP, ns_hist))
        # print("P:", P)
        P = P * theta_old

        theta = np.copy(P / sum(P))
        # print("hist:", ns_hist)
        # print("theta:", theta)

        # logliklihood = np.inner(ns_hist, np.log(np.matmul(transform, theta)))
        # imporve = logliklihood - old_logliklihood

        # if r > 1 and abs(imporve) < loglikelihood_threshold:
        #     print("stop when", imporve, loglikelihood_threshold)
        #     break

        # old_logliklihood = logliklihood
        # print("logliklihood", logliklihood)
        # print("r", r)
        r += 1
        end_time = time.time()
        # print("running time each iter", end_time - begin_time)
    return theta

def EMS(n, ns_hist, transform, max_iteration, loglikelihood_threshold, init_dist):
    # smoothing matrix
    smoothing_factor = 2
    binomial_tmp = [scipy.special.binom(smoothing_factor, k) for k in range(smoothing_factor + 1)]
    smoothing_matrix = np.zeros((n, n))
    central_idx = int(len(binomial_tmp) / 2)
    for i in range(int(smoothing_factor / 2)):
        smoothing_matrix[i, : central_idx + i + 1] = binomial_tmp[central_idx - i:]
    for i in range(int(smoothing_factor / 2), n - int(smoothing_factor / 2)):
        smoothing_matrix[i, i - central_idx: i + central_idx + 1] = binomial_tmp
    for i in range(n - int(smoothing_factor / 2), n):
        remain = n - i - 1
        smoothing_matrix[i, i - central_idx + 1:] = binomial_tmp[: central_idx + remain]
    row_sum = np.sum(smoothing_matrix, axis=1)
    smoothing_matrix = (smoothing_matrix.T / row_sum).T

    smoothing_matrix[-1, -1] = 2/3
    smoothing_matrix[-1, -2] = 1/3
    # print("smoothing_matrix", smoothing_matrix)

    # EMS
    theta = np.ones(n) / float(n)
    theta_old = np.zeros(n)
    r = 0
    sample_size = sum(ns_hist)
    old_logliklihood = 0

    # while LA.norm(theta_old - theta, ord=1) > 1 / sample_size and r < max_iteration:
    # while r < max_iteration:
    while (LA.norm(theta_old - theta, ord=1) > 1 / sample_size) and (r < max_iteration):
        theta_old = np.copy(theta)
        X_condition = np.matmul(transform, theta_old)

        TMP = transform.T / X_condition

        P = np.copy(np.matmul(TMP, ns_hist))
        P = P * theta_old

        theta = np.copy(P / sum(P))

        # Smoothing step
        theta = np.matmul(smoothing_matrix, theta)
        theta = theta / sum(theta)

        logliklihood = np.inner(ns_hist, np.log(np.matmul(transform, theta)))
        imporve = logliklihood - old_logliklihood

        # if r > 1 and abs(imporve) < loglikelihood_threshold:
            # print("stop when", imporve / old_logliklihood, loglikelihood_threshold)
            # break

        old_logliklihood = logliklihood

        r += 1
    return theta

def loglikelihood_calc(random_matrix, est_dist, noisy_result_dist):
    # print("est_dist", est_dist)
    # print("noisy_result_dist", noisy_result_dist)
    # loglikelihood = 0
    # for i in range(len(noisy_result_dist)):
    #     loglikelihood += noisy_result_dist[i] * np.log(np.dot(random_matrix[i], est_dist))
    # normalized_est_dist = est_dist / np.sum(est_dist)
    # normalized_noisy_result_dist = noisy_result_dist / np.sum(noisy_result_dist)
    loglikelihood = np.inner(noisy_result_dist, np.log(np.matmul(random_matrix, est_dist)))
    return loglikelihood


def gen_random_matrix_grr(eps, domain):
    random_matrix = np.zeros([domain, domain])
    p = np.exp(eps) / (np.exp(eps) + domain - 1)
    q = 1 / (np.exp(eps) + domain - 1)
    for i in range(random_matrix.shape[0]):
        random_matrix[i] = q
        random_matrix[i, i] = p
    return random_matrix


def gen_random_matrix_oue(eps, domain):
    random_matrix = np.zeros([domain, domain])
    p = 1/2
    q = 1/(np.exp(eps) + 1)

    for i in range(random_matrix.shape[0]):
        random_matrix[i] = q
        random_matrix[i, i] = p
    return random_matrix

# def gen_random_matrix_oue(eps, domain):   # random matrix for OUE with sampling
#     random_matrix = np.zeros([domain, domain])
#     p = 1 / 2
#     q = 1 / (np.exp(eps) + 1)
#
#     for i in range(random_matrix.shape[0]):
#         prob = 0
#         for l in range(2, domain + 1):
#             prob += (q * comb(domain-2, l-2) * p * q ** (l - 2) * (1 - q) ** (domain - l)) / l
#         for l in range(1, domain):
#             prob += (q * comb(domain-2, l-1) * (1 - p) * q ** (l - 1) * (1 - q) ** (domain - l - 1)) / l
#         random_matrix[i] = prob
#
#         prob = 0
#         for l in range(1, domain+1):
#             prob += (p * comb(domain-1, l-1) * q**(l-1) * (1-q)**(domain-l)) / l
#         random_matrix[i, i] = prob
#     return random_matrix

def gen_random_matrix_olh(eps, domain):
    # g = int(round(np.exp(eps))) + 1
    g = round(np.exp(eps)) + 1
    random_matrix = np.zeros([domain, domain])
    ave_num_per_hash = domain / g
    p = np.exp(eps) / (np.exp(eps) + g - 1)
    q = 1 / (np.exp(eps) + g - 1)

    for i in range(random_matrix.shape[0]):
        random_matrix[i] = 1/g
        random_matrix[i, i] = p
    return random_matrix


# def gen_random_matrix_olh(eps, domain): # random matrix for OLH with sampling
#     g = int(round(np.exp(eps))) + 1
#     random_matrix = np.zeros([domain, domain])
#     ave_num_per_hash = domain / g
#     p = np.exp(eps) / (np.exp(eps) + g - 1)
#     q = 1 / (np.exp(eps) + g - 1)
#
#     for i in range(random_matrix.shape[0]):
#         random_matrix[i] = 1/g * p / ave_num_per_hash + (g-1) / g * q / ave_num_per_hash
#         random_matrix[i, i] = p / ave_num_per_hash
#     return random_matrix



def gen_random_matrix_pm(epsilon, m):
    s = (np.exp(epsilon / 2) + 1) / (np.exp(epsilon / 2) - 1)
    high_prob_density = (np.exp(epsilon / 2) / 2) * ((np.exp(epsilon / 2) - 1) / (np.exp(epsilon / 2) + 1))
    low_prob_density = (1 / (2 * np.exp(epsilon / 2))) * ((np.exp(epsilon / 2) - 1) / (np.exp(epsilon / 2) + 1))

    # 定义输入和输出的bins
    input_bins = np.linspace(-1, 1, m + 1)
    output_bins = np.linspace(-s, s, m + 1)

    # 初始化概率转移矩阵M
    M = np.zeros((m, m))

    # 计算输入桶的宽度和输出桶的宽度
    input_bin_width = 2 / m
    output_bin_width = 2 * s / m

    for i in range(m):
        # 计算第i个输入桶的中心值v
        v = (input_bins[i] + input_bins[i + 1]) / 2
        l_v = (np.exp(epsilon / 2) * v - 1) / (np.exp(epsilon / 2) - 1)
        r_v = (np.exp(epsilon / 2) * v + 1) / (np.exp(epsilon / 2) - 1)

        # 计算概率
        for j in range(m):
            bin_start = output_bins[j]
            bin_end = output_bins[j + 1]

            # 高概率区域与当前桶的重叠部分
            high_start = max(l_v, bin_start)
            high_end = min(r_v, bin_end)
            if high_end > high_start:
                # 只有当高概率区域实际与输出桶有重叠时才计算
                overlap = high_end - high_start
                M[j, i] += high_prob_density * overlap# / output_bin_width
            # 低概率区域
            # 每个输出桶除了高概率区域以外都应该加上低概率密度乘以剩余的部分
            if bin_start < l_v:
                low_overlap = min(l_v, bin_end) - bin_start
                M[j, i] += low_prob_density * low_overlap# / output_bin_width
            if bin_end > r_v:
                low_overlap = bin_end - max(r_v, bin_start)
                M[j, i] += low_prob_density * low_overlap# / output_bin_width

    # 归一化每一列，确保概率守恒
    # M /= np.sum(M, axis=0)

    return M

def gen_random_matrix_sw(eps, m):
    M, _, _ = sw_compromised.sw(np.random.uniform(size=10000), 0, 1, eps, [], 0, 0, m, m)
    return M

def attack_grr(data, eps, domain, r, beta):
    n = np.size(data)
    noisy_result_f = grr.grr_perturbation(data, eps, domain)
    target_item_index_f = np.random.choice(domain - 1, size=r, replace=False)
    fake_user_num_f = int(beta * n)
    if beta > 0:
        # fake_value_f = np.zeros(fake_user_num_f)
        # fake_value_f += target_item_index_f
        fake_value_f = np.random.choice(target_item_index_f, size = fake_user_num_f, replace=True)
        noisy_result_f = np.concatenate((noisy_result_f[:-fake_user_num_f], fake_value_f), axis=0)
    noisy_result_dist_f, _ = np.histogram(noisy_result_f, bins=domain)

    ldp_est_dist = grr.grr_aggregation(noisy_result_f, eps, domain, n)
    ldp_est_dist, _ = norm_sub(n, ldp_est_dist)
    ldp_est_dist = ldp_est_dist / n

    # import matplotlib.pyplot as plt
    # data_dist, _ = np.histogram(syn_data, bins=domain)
    # data_dist = data_dist / n
    # plt.bar(np.arange(domain), ldp_est_dist)
    # plt.title("syn data distribution")
    # plt.show()

    return ldp_est_dist, noisy_result_f, noisy_result_dist_f



def attack_oue(data, eps, domain, r, beta):
    n = np.size(data)
    noisy_result_f = oue.oue_perturb(data, eps, domain)
    target_item_index_f = np.random.choice(domain-1, size = r, replace = False)
    fake_user_num_f = int(beta * n)
    p = 1/2
    q = 1/(np.exp(eps) + 1)
    l = int((p + (domain - 1) * q - 1))
    if beta > 0:
        fake_value_f = np.zeros([fake_user_num_f, domain])
        fake_value_f[:, target_item_index_f] = 1
        for k in range(fake_user_num_f):
            if int(l - r) < 0:
                break
            pos = np.random.choice(np.arange(domain), int(l - r), replace=False)
            fake_value_f[k, pos] = 1
        noisy_result_f = np.concatenate((noisy_result_f[:-fake_user_num_f], fake_value_f), axis=0)

    ldp_est_dist = oue.oue_aggregator(noisy_result_f, eps, domain)
    ldp_est_dist, _ = norm_sub(n, ldp_est_dist)
    # ldp_est_dist = mle_apx(eps, n, ldp_est_dist)
    ldp_est_dist = ldp_est_dist / n
    noisy_result_dist_f = np.sum(noisy_result_f, axis = 0)  # ns without sampling


    #############################################
    # temp_matrix = []    # ns with sampling
    # for row in range(noisy_result_f.shape[0]):
    #     temp_row = noisy_result_f[row]
    #     mark_one = np.where(temp_row == 1)[0]
    #     sample = np.random.choice(mark_one, 1)[0]
    #     vector = [0] * domain
    #     vector[sample] = 1
    #     temp_matrix.append(vector)
    # temp_matrix = np.array(temp_matrix)
    # noisy_result_dist_f = np.sum(temp_matrix, axis=0) / n
    #############################################

    # noisy_result_dist_f = np.sum(noisy_result_f, axis=0) / np.sum(noisy_result_f)  # ns with dummy user expansion


    return ldp_est_dist, noisy_result_f, noisy_result_dist_f


def attack_olh_user(data, eps, domain, r, beta):
    n = np.size(data)
    ldp_est_dist, noisy_result_f, noisy_result_dist_f = OLH_compromised_user.olh(data, eps, domain, r, beta, 0)
    ldp_est_dist, _ = norm_sub(n, ldp_est_dist)
    ldp_est_dist = ldp_est_dist / n
    return ldp_est_dist, noisy_result_f, noisy_result_dist_f

def attack_olh_server(data, eps, domain, r, beta):
    n = np.size(data)
    ldp_est_dist, noisy_result_f, noisy_result_dist_f = OLH_compromised_server.olh(data, eps, domain, r, beta, 0)
    ldp_est_dist, _ = norm_sub(n, ldp_est_dist)
    ldp_est_dist = ldp_est_dist / n
    return ldp_est_dist, noisy_result_f, noisy_result_dist_f

def attack_pm(data, eps, domain, beta, true_mean, attack_type):
    n = np.size(data)
    fake_user_num_f = int(beta * n)
    s = (np.exp(eps / 2) + 1) / (np.exp(eps / 2) - 1)
    if beta > 0:
        if attack_type == 0:
            fake_value_f = np.ones(fake_user_num_f) * s
        if attack_type == 1:
            fake_value_f = np.random.uniform(1, s, size=fake_user_num_f)
        if attack_type == 2:
            fake_value_f = np.random.uniform(1/2 * s, s, size=fake_user_num_f)
        if attack_type == 3:
            fake_value_f = np.random.uniform(true_mean, s, size=fake_user_num_f)
            # fake_value_f = np.ones(fake_user_num_f) * s
        if attack_type == 4:    # fine-grained attack
            true_sum_f = np.sum(data)
            target_mean_f = 0.62
            fake_value_f = np.ones(fake_user_num_f) * (n*target_mean_f - true_sum_f)
            fake_value_f = fake_value_f + np.random.normal(0, 0.001, size=fake_value_f.shape)

    else:
        fake_value_f = np.array([])
    noisy_result_f, _ = compromised_pm.estimate_mean(data, fake_value_f, eps)
    noisy_result_dist_f, _ = np.histogram(noisy_result_f, bins=domain)
    # print("noisy_result_dist_f", noisy_result_dist_f)
    # plt.bar(np.arange(domain), noisy_result_dist_f)
    # plt.title("noisy_result_dist_f")
    # plt.show()
    return noisy_result_f, noisy_result_dist_f

def attack_sw(data, eps, domain, beta, target_value, attack_type):
    l = np.min(data)
    h = np.max(data)
    n = np.size(data)
    fake_user_num = int(beta * n)
    _, noisy_result_f, noisy_result_dist_f = sw_compromised.sw(data, l, h, eps, target_value, fake_user_num, attack_type, randomized_bins=domain, domain_bins=domain)

    return noisy_result_f, noisy_result_dist_f

def identify_attack_by_3_sigma(chain):
    if adfuller(chain)[1] > 0.05:
        print("attack detected by 3std")
        return 1
    for i in range(1, len(chain)):
        if adfuller(chain[:-i])[1] < 0.05:
            # print("p value:", adfuller(chain[:-i])[1])
            continue
        else:
            # print("p value:", adfuller(chain[-i:])[1])
            sub_chain = chain[-i:]
            break
    print("chain", chain)
    print("sub chain", sub_chain)
    mean = np.mean(sub_chain)
    std = np.std(sub_chain)
    print("std", std)
    print("diff", np.fabs(chain[0] - mean))
    if mean - 3 * std < chain[0] < mean + 3 * std:
        print("no attack detected by 3std")
        return 0
    else:
        print("attack detected by 3std")
        return 1

def identify_attack_STL_decompose(chain):
    chain = np.array(chain)
    trend = mk.regional_test(np.array(chain), alpha=0.002)[0]
    # print(trend)
    if trend == "no trend":
        result = STL(chain, period=len(chain), seasonal=len(chain)).fit()
        resid = np.fabs(chain - result.trend)
        if np.argmax(resid) == 0:
            print("attack detected by 3std")
            return 1
        else:
            print("no attack detected by 3std")
            return 0
    else:
        print("attack detected by 3std")
        return 1

def identify_attack(chain):
    chain = np.array(chain)
    trend = mk.regional_test(np.array(chain), alpha=0.002)[0]
    # print(trend)
    if trend == "no trend":
        # result = STL(chain, period=len(chain), seasonal=len(chain)).fit()
        resid = np.fabs(detrend(chain))
        # import matplotlib.pyplot as plt
        # plt.plot(np.arange(len(resid)), resid)
        # plt.title("residual")
        # plt.show()
        if np.argmax(resid) == 0:
            print("attack detected by 3std")
            return 1
        else:
            print("no attack detected by 3std")
            return 0
    else:
        print("attack detected by 3std")
        return 1

def likelihood_chain_gen(eps, n, max_iter, ldp_est_dist, random_matrix, reported_noisy_result_dist, chain_length, domain, ldp_protocol, true_mean, attack_type):
    ave_num = 2
    chain = []
    est_dist_list = []
    loglikelihood = 0
    for _ in range(1):
        est_dist = EM(domain, reported_noisy_result_dist, random_matrix, max_iter, 1e-3, ldp_est_dist)
        est_dist_list.append(est_dist)
        loglikelihood += loglikelihood_calc(random_matrix, est_dist, reported_noisy_result_dist)
    loglikelihood = loglikelihood / 1
    est_dist = np.mean(est_dist_list, axis=0)
    # import matplotlib.pyplot as plt
    # plt.bar(np.arange(domain), est_dist)
    # plt.title("est_dist")
    # plt.show()
    first_est_dist = np.copy(est_dist)
    chain.append(loglikelihood)
    # print("loglikelihood", loglikelihood)

    #-----------------------------------------------------------------------------------------------------------------------
    # if ldp_protocol == 0 or ldp_protocol == 1 or ldp_protocol == 2: # generate the last noisy report distribution to compare the last generated "noisy_result_dist"
    #     est_dist_syn = synthesize_discrete_data(n, est_dist, domain)
    #     if ldp_protocol == 0:
    #         _, _, noisy_result_dist_from_report = attack_oue(est_dist_syn, eps, domain, 0, 0)
    #     if ldp_protocol == 1:
    #         _, _, noisy_result_dist_from_report = attack_olh_user(est_dist_syn, eps, domain, 0, 0)
    #     if ldp_protocol == 2:
    #         _, _, noisy_result_dist_from_report = attack_olh_server(est_dist_syn, eps, domain, 0, 0)
    # else:
    #     est_dist_syn = synthesize_numerical_data(n, est_dist)
    #     _, noisy_result_dist_from_report = attack_pm(est_dist_syn, eps, domain, 0, true_mean, attack_type)
    #------------------------------------------------------------------------------------------------------------------------


    if ldp_protocol == 0 or ldp_protocol == 1 or ldp_protocol == 2 or ldp_protocol == 3:
        syn_data = synthesize_discrete_data(n, est_dist, domain)
        # syn_data_hist, _ = np.histogram(syn_data, bins = domain)
        # plt.bar(np.arange(domain), syn_data_hist)
        # plt.title("syn_data_hist")
        # plt.show()
    else:
        syn_data = synthesize_numerical_data(n, est_dist)

    # import matplotlib.pyplot as plt
    # data_dist, _ = np.histogram(syn_data, bins=domain)
    # data_dist = data_dist / n
    #
    # plt.bar(np.arange(domain), data_dist)
    # plt.title("syn data distribution")
    # plt.show()

    for c in range(chain_length):
        # print("chain length", c)
        noisy_result_dist_list = []
        est_dist_list = []
        loglikelihood = 0
        for _ in range(ave_num):
            if ldp_protocol == 0:
                ldp_est_dist, noisy_result, noisy_result_dist = attack_grr(syn_data, eps, domain, 0, 0)
            if ldp_protocol == 1:
                ldp_est_dist, noisy_result, noisy_result_dist = attack_oue(syn_data, eps, domain, 0, 0)
            if ldp_protocol == 2:
                ldp_est_dist, noisy_result, noisy_result_dist = attack_olh_user(syn_data, eps, domain, 0, 0)
            if ldp_protocol == 3:
                ldp_est_dist, noisy_result, noisy_result_dist = attack_olh_server(syn_data, eps, domain, 0, 0)
            if ldp_protocol == 4:
                noisy_result, noisy_result_dist = attack_pm(syn_data, eps, domain, 0, true_mean, attack_type)
            if ldp_protocol == 5:
                noisy_result, noisy_result_dist = attack_sw(syn_data, eps, domain, 0, [], attack_type)

            est_dist = EM(domain, noisy_result_dist, random_matrix, max_iter, 1e-3, ldp_est_dist)    # EM is used for PM and SW
            est_dist_list.append(est_dist)
            noisy_result_dist_list.append(noisy_result_dist)

            loglikelihood += loglikelihood_calc(random_matrix, est_dist, noisy_result_dist)
        # noisy_result_dist = np.mean(noisy_result_dist_list, axis=0)
        est_dist = np.mean(est_dist_list, axis=0)
        # import matplotlib.pyplot as plt
        # plt.bar(np.arange(domain), est_dist)
        # plt.title("EM aggregated result")
        # plt.show()
        loglikelihood = loglikelihood / ave_num
        chain.append(loglikelihood)
        # print("loglikelihood", loglikelihood)

        if ldp_protocol == 0:
            syn_data = synthesize_discrete_data(n, est_dist, domain)
            _, _, noisy_result_dist =  attack_grr(syn_data, eps, domain, 0, 0)

        if ldp_protocol == 1:
            syn_data = synthesize_discrete_data(n, est_dist, domain)
            _, _, noisy_result_dist = attack_oue(syn_data, eps, domain, 0, 0)

        # compare synthetic data with the first est dist-----------------------------------------
        # if ldp_protocol == 0 or ldp_protocol == 1 or ldp_protocol == 2 or ldp_protocol == 3:
        #     syn_data = synthesize_discrete_data(n*5, est_dist, domain)
        #     first_syn_data = synthesize_discrete_data(n*2, first_est_dist, domain)
        #     p_value = stats.kstest(syn_data, first_syn_data)[1]
        #     if p_value > 0.05:
        #         est_dist = first_est_dist



        #------------------------------------------------------------------------------------------


        # compare noisy report distribution with the first generated "noisy_result_dist"---------------------------------------------------------------------------------------------------
        if ldp_protocol == 0 or ldp_protocol == 1 or ldp_protocol == 2 or ldp_protocol == 3:
            # reported_noisy_result_dist_syn = synthesize_discrete_data(int(np.sum(reported_noisy_result_dist)), reported_noisy_result_dist / np.sum(reported_noisy_result_dist), domain)
            # noisy_result_dist_syn = synthesize_discrete_data(int(np.sum(noisy_result_dist)), noisy_result_dist / np.sum(noisy_result_dist), domain)
            # p_value = stats.kstest(reported_noisy_result_dist_syn, noisy_result_dist_syn)[1]
            # print("p_value", p_value)
            #
            # if p_value > 0.05:
            #     est_dist = first_est_dist
            # syn_data = synthesize_discrete_data(n, est_dist, domain)

            # print("reported_noisy_result_dist", reported_noisy_result_dist / np.sum(reported_noisy_result_dist) * n)
            # print("noisy_result_dist", noisy_result_dist / np.sum(noisy_result_dist) * n)

            p_value = stats.chisquare(noisy_result_dist / np.sum(noisy_result_dist) * n, reported_noisy_result_dist / np.sum(reported_noisy_result_dist) * n)[1]
            # print("p_value", p_value)

            if p_value >= 0.0001:
                est_dist = first_est_dist
            syn_data = synthesize_discrete_data(n, est_dist, domain)


        else:
            reported_noisy_result_dist_syn = synthesize_numerical_data(n, reported_noisy_result_dist / np.sum(reported_noisy_result_dist))
            noisy_result_dist_syn = synthesize_numerical_data(n, noisy_result_dist / np.sum(noisy_result_dist))
            p_value = stats.kstest(reported_noisy_result_dist_syn, noisy_result_dist_syn)[1]
            # print("p_value", p_value)

            if p_value > 0.05:
                est_dist = first_est_dist
            # print("first_est_dist", first_est_dist)
            syn_data = synthesize_numerical_data(n, est_dist)

        # if c % 10 == 0:
        #     import matplotlib.pyplot as plt
        #     data_dist, _ = np.histogram(syn_data, bins=domain)
        #     data_dist = data_dist / n
        #
        #     plt.bar(np.arange(domain), data_dist)
        #     plt.title("syn data distribution")
        #     plt.show()
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------
        # if ldp_protocol == 0 or ldp_protocol == 1 or ldp_protocol == 2: # compare noisy report distribution with the last generated "noisy_result_dist"
        #     est_dist_syn = synthesize_discrete_data(n, est_dist, domain)
        #     if ldp_protocol == 0:
        #         _, _, noisy_result_dist = attack_oue(est_dist_syn, eps, domain, 0, 0)
        #     if ldp_protocol == 1:
        #         _, _, noisy_result_dist = attack_olh_user(est_dist_syn, eps, domain, 0, 0)
        #     if ldp_protocol == 2:
        #         _, _, noisy_result_dist = attack_olh_server(est_dist_syn, eps, domain, 0, 0)
        #
        #     reported_noisy_result_dist_syn = synthesize_discrete_data(int(np.sum(noisy_result_dist_from_report)), noisy_result_dist_from_report / np.sum(noisy_result_dist_from_report), domain)
        #     noisy_result_dist_syn = synthesize_discrete_data(int(np.sum(noisy_result_dist)), noisy_result_dist / np.sum(noisy_result_dist), domain)
        #     p_value = stats.kstest(reported_noisy_result_dist_syn, noisy_result_dist_syn)[1]
        #     # print("p_value", p_value)
        #
        #     if p_value > 0.05:
        #         est_dist = first_est_dist
        #     syn_data = synthesize_discrete_data(n, est_dist, domain)
        # else:
        #     est_dist_syn = synthesize_numerical_data(n, est_dist)
        #     _, noisy_result_dist = attack_pm(est_dist_syn, eps, domain, 0, true_mean, attack_type)
        #     reported_noisy_result_dist_syn = synthesize_numerical_data(n, noisy_result_dist_from_report / np.sum(noisy_result_dist_from_report))
        #     noisy_result_dist_syn = synthesize_numerical_data(n, noisy_result_dist / np.sum(noisy_result_dist))
        #     p_value = stats.kstest(reported_noisy_result_dist_syn, noisy_result_dist_syn)[1]
        #     # print("p_value", p_value)
        #
        #     if p_value > 0.05:
        #         est_dist = first_est_dist
        #     syn_data = synthesize_numerical_data(n, est_dist)
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------

    return chain