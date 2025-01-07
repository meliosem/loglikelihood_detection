import numpy as np
import xxhash
import csv

def calc_mean(dist, bin, size):
    bin_width = 1 / bin
    sampled_data = np.array([])
    for i in range(len(dist)):
        if dist[i] >= 0:
            sampled_data = np.append(sampled_data, np.random.uniform(i * bin_width, (i + 1) * bin_width, int(size * dist[i])))
        else:
            sampled_data = np.append(sampled_data, -1 * np.random.uniform(i * bin_width, (i + 1) * bin_width, int(size * -dist[i])))
    return np.mean(sampled_data)


def olh(ori_samples, eps, domain, r, beta, post_type):
    # samples = (ori_samples - l) / (h - l)
    samples = np.copy(ori_samples)
    np.random.shuffle(samples)
    n = np.size(samples)

    # print("n", n)
    # import matplotlib.pyplot as plt
    # syn_data_hist, _ = np.histogram(samples, bins=domain)
    # plt.bar(np.arange(domain), syn_data_hist)
    # plt.title("syn_data_hist in fun")
    # plt.show()

    g = int(round(np.exp(eps))) + 1
    p = np.exp(eps) / (np.exp(eps) + g - 1)
    q = 1.0 / (np.exp(eps) + g - 1)
    noisy_samples = np.zeros([np.size(samples)])

    # encode
    perturb_matrix = np.random.randint(0, g, size=(n, domain))
    # np.save("perturb_matrix_server.npy", perturb_matrix)
    # perturb_matrix = np.load("./perturb_matrix_server.npy")
    projected_index = np.array([perturb_matrix[i, samples[i]] for i in range(n)])

    # perturb
    randoms = np.random.random(size=np.size(projected_index))
    mark_perturb = randoms > p - q
    num_perturb = np.sum(mark_perturb)
    perturb_result = np.random.randint(g, size=num_perturb)
    projected_index[mark_perturb] = perturb_result

    if beta > 0:
        fake_user_num = int(n * beta)
        target_item = np.random.choice(np.arange(domain-20, domain - 1), size=r, replace=False)
        fake_value_list = []
        for i in range(fake_user_num):
            projected_value_by_target = perturb_matrix[-(fake_user_num-i), target_item]
            fake_index = np.argmax(np.bincount(projected_value_by_target))
            fake_value_list.append(fake_index)
        projected_index[-fake_user_num:] = fake_value_list

    if post_type == 0:
        # aggregation
        noisy_result_dist = np.zeros(domain)
        for i in range(n):
            temp_vector = np.zeros(domain)
            temp_vector[perturb_matrix[i] == projected_index[i]] = 1
            noisy_result_dist += temp_vector

        a = 1.0 * g / (p * g - 1)
        b = 1.0 * n / (p * g - 1)
        result = a * noisy_result_dist - b
        result = (noisy_result_dist - n * 1./g) / (p - 1./g)
        return result, noisy_samples, noisy_result_dist



def norm_sub(n, noisy_samples, hash_seed, p, g, domain_bins):  # this is count, not frequency
    print("norm_sub")
    result = np.zeros(domain_bins)
    for i in range(n):
            for v in range(domain_bins):
                if noisy_samples[i] == (xxhash.xxh32(str(v), seed=hash_seed[i]).intdigest() % g):
                    result[v] += 1

    a = 1.0 * g / (p * g - 1)
    b = 1.0 * n / (p * g - 1)
    result = a * result - b

    while (np.fabs(sum(result) - n) > 1) or (result < 0).any(): # Norm-Sub
        result[result < 0] = 0
        total = sum(result)
        mask = result > 0
        diff = (n - total) / sum(mask)
        result[mask] += diff

    return result, n

# def norm_sub(n, noisy_samples, fake_user_num, optimal_seed_array, p, g, domain_bins):  # this is count, not frequency
#     print("norm_sub")
#     result = np.zeros(domain_bins)
#     if np.size(optimal_seed_array) != 0:
#         for i in range(n):
#             if i < n - fake_user_num:
#                 for v in range(domain_bins):
#                     if noisy_samples[i] == (xxhash.xxh32(str(v), seed=i).intdigest() % g):
#                         result[v] += 1
#             else:
#                 for v in range(domain_bins):
#                     if noisy_samples[i] == (xxhash.xxh32(str(v), seed=optimal_seed_array[i - (n-fake_user_num)]).intdigest() % g):
#                         result[v] += 1
#     else:
#         for i in range(n):
#             for v in range(domain_bins):
#                 if noisy_samples[i] == (xxhash.xxh32(str(v), seed=i).intdigest() % g):
#                     result[v] += 1
#
#     a = 1.0 * g / (p * g - 1)
#     b = 1.0 * n / (p * g - 1)
#     result = a * result - b
#
#     while (np.fabs(sum(result) - n) > 1) or (result < 0).any(): # Norm-Sub
#         result[result < 0] = 0
#         total = sum(result)
#         mask = result > 0
#         diff = (n - total) / sum(mask)
#         result[mask] += diff
#
#     return result, n

def norm(n, noisy_samples, hash_seed, p, g, domain_bins):   # this is count, not frequency
    print("norm")
    result = np.zeros(domain_bins)
    for i in range(n):
        for v in range(domain_bins):
            if noisy_samples[i] == (xxhash.xxh32(str(v), seed=hash_seed[i]).intdigest() % g):
                result[v] += 1
    a = 1.0 * g / (p * g - 1)
    b = 1.0 * n / (p * g - 1)
    result = a * result - b
    estimates = np.copy(result)
    total = sum(estimates)
    domain_size = len(result)
    diff = (n - total) / domain_size
    estimates += diff
    return estimates, n

def norm_mul(n, noisy_samples, hash_seed, p, g, domain_bins):
    print("norm_mul")
    result = np.zeros(domain_bins)
    for i in range(n):
        for v in range(domain_bins):
            if noisy_samples[i] == (xxhash.xxh32(str(v), seed=hash_seed[i]).intdigest() % g):
                result[v] += 1
    a = 1.0 * g / (p * g - 1)
    b = 1.0 * n / (p * g - 1)
    result = a * result - b
    estimates = np.copy(result)
    estimates[estimates < 0] = 0
    total = sum(estimates)
    return estimates * n / total, n

def norm_cut(n, noisy_samples, hash_seed, p, g, domain_bins):
    print("norm_cut")
    result = np.zeros(domain_bins)
    for i in range(n):
        for v in range(domain_bins):
            if noisy_samples[i] == (xxhash.xxh32(str(v), seed=hash_seed[i]).intdigest() % g):
                result[v] += 1
    a = 1.0 * g / (p * g - 1)
    b = 1.0 * n / (p * g - 1)
    result = a * result - b
    estimates = np.copy(result)
    order_index = np.argsort(estimates)

    total = 0
    for i in range(len(order_index)):
        total += estimates[order_index[- 1 - i]]
        if total > n:
            break
        if total < n and np.abs(-2-i) == len(order_index):
            return estimates, n
        if total < n and estimates[order_index[- 2 - i]] <= 0:
            estimates[order_index[:- 1 - i]] = 0
            return estimates * n / total, n

    for j in range(i + 1, len(order_index)):
        estimates[order_index[- 1 - j]] = 0

    return estimates * n / total, n

# test OLH code
if __name__ == "__main__":
    # data = np.load("../data/normal_numerical.npy")
    # data = np.load("../data/taxi_numerical.npy")
    # data = np.load("../data/retirement_numerical.npy")
    data = np.load("../data/zipf.npy")
    data = data[data < 80]
    n = np.size(data)
    print("size:", n)

    # l = np.min(data)
    # h = np.max(data)
    # true_mean = np.mean((data - l) / (h - l))
    # eps = 1
    # attack_type = 0
    # post_type = 1
    # bins = 32

    eps = 1
    target_value = np.array([10, 20, 5])
    r = 5
    beta = 0.05
    domain = np.max(data) + 1
    post_type = 0

    import matplotlib.pyplot as plt

    data_dist, _ = np.histogram(data, bins=domain)
    data_dist = data_dist / n
    plt.bar(np.arange(domain), data_dist)
    plt.title("data distribution")
    plt.show()

    result, noisy_result, _ = olh(data, eps, domain, r, beta, post_type)

    est_dist, _ = np.histogram(result, bins=domain)
    est_dist = est_dist / n
    plt.bar(np.arange(domain), result / n)
    plt.title("est distribution")
    plt.show()