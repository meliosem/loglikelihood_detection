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

# def olh(ori_samples, l, h, eps, target_value, fake_user_num, attack_type, post_type, domain_bins):
#     samples = (ori_samples - l) / (h - l)
#     np.random.shuffle(samples)
#     n = np.size(samples)
#     bin_width = 1 / domain_bins
#     index = samples / bin_width
#     index = index.astype("int")
#     mark = index == domain_bins
#     index[mark] -= 1
#
#     hash_num = 1000
#     # g = int(round(np.exp(eps))) + 1
#     g = int(np.exp(eps)) + 1
#     p = np.exp(eps) / (np.exp(eps) + g - 1)
#     q = 1.0 / (np.exp(eps) + g - 1)
#     noisy_samples = np.zeros([np.size(samples)])
#
#     for i in range(n):
#         v = index[i]
#         x = (xxhash.xxh32(str(v), seed=i).intdigest() % g)
#         y = x
#
#         p_sample = np.random.random_sample()
#         if p_sample > p - q:
#             # perturb
#             y = np.random.randint(0, g)
#         noisy_samples[i] = y
#
#     # fake value
#     optimal_seed_array = np.array([])
#     if np.size(target_value) != 0:
#         print("eps:", eps)
#         if attack_type == 0:
#             print("attack on one bin OLHB")
#             hash_range = 0
#             optimal_seed_array = np.array([])
#             optimal_fake_v_array = np.array([])
#             for _ in range(fake_user_num):
#                 optimal_seed = 0
#                 optimal_fake_v = 0
#                 max_ave = 0
#                 for hash_seed in range(hash_range, hash_num + hash_range):   # search fake value and hash function
#                     for fake_v in range(g):
#                         s = 0
#                         count = 0
#                         for v in range(domain_bins):
#                             if fake_v == (xxhash.xxh32(str(v), seed=hash_seed).intdigest() % g):
#                                 s += v
#                                 count += 1
#                         if count == 0:
#                             ave = 0
#                             if ave > max_ave:
#                                 max_ave = ave
#                                 optimal_seed = hash_seed
#                                 optimal_fake_v = fake_v
#                         else:
#                             ave = s / count
#                             if ave > max_ave:
#                                 max_ave = ave
#                                 optimal_seed = hash_seed
#                                 optimal_fake_v = fake_v
#                 optimal_seed_array = np.append(optimal_seed_array, optimal_seed)
#                 optimal_fake_v_array = np.append(optimal_fake_v_array, optimal_fake_v)
#                 hash_range += hash_num
#
#             optimal_seed_array = optimal_seed_array.astype("int")
#             optimal_fake_v_array = optimal_fake_v_array.astype("int")
#
#             np.save("optimal_seed_array_g_" + str(g) + ".npy", optimal_seed_array)
#             np.save("optimal_fake_v_array_g_" + str(g) + ".npy", optimal_fake_v_array)
#
#             if target_value[0] == 1:  # maximizing mean
#                 fake_value = np.zeros([fake_user_num]) + optimal_fake_v_array
#
#                 # for fake_v in range(g):
#                 #     if fake_v == (xxhash.xxh32(str(domain_bins - 1), seed=i).intdigest() % g):
#                 #         fake_value[i - (n-fake_user_num)] = fake_v
#
#         # print("noisy:", noisy_samples)
#         # print("fake:", fake_value)
#         # print("seed:", optimal_seed_array)
#         noisy_samples = np.concatenate( (noisy_samples[:-fake_user_num], fake_value), axis = 0 )
#         # print(noisy_samples[-1000:] == fake_value)
#
#     # print("fake_value", fake_value)
#     if post_type == 0:
#         result = np.zeros(domain_bins)
#         for i in range(n):
#             for v in range(domain_bins):
#                 if noisy_samples[i] == (xxhash.xxh32(str(v), seed=i).intdigest() % g):
#                     result[v] += 1
#         a = 1.0 * g / (p * g - 1)
#         b = 1.0 * n / (p * g - 1)
#         result = a * result - b
#         return result, n
#     if post_type == 1:
#         return norm_sub(n, noisy_samples, fake_user_num, optimal_seed_array, p, g, domain_bins)
#     if post_type == 2:
#         return norm(n, noisy_samples, p, g, domain_bins)
#     if post_type == 3:
#         return norm_mul(n, noisy_samples, p, g, domain_bins)
#     if post_type == 4:
#         return norm_cut(n, noisy_samples, p, g, domain_bins)

def olh(ori_samples, eps, domain, r, beta, post_type):
    # if eps == 0.2 or eps == 0.6:
    #     fake_value_array = np.load("fake_value_array_eps02.npy")
    # if eps == 1:
    #     fake_value_array = np.load("fake_value_array_eps10.npy")

    # samples = (ori_samples - l) / (h - l)
    samples = np.copy(ori_samples)
    np.random.shuffle(samples)
    n = np.size(samples)
    # bin_width = 1 / domain
    # index = samples / bin_width
    # index = index.astype("int")
    # mark = index == domain
    # index[mark] -= 1

    hash_seed = np.random.choice(n, size=n, replace=False)
    g = int(round(np.exp(eps))) + 1
    # g = 16   # test robustness with different g
    # g = int(np.exp(eps)) + 1
    p = np.exp(eps) / (np.exp(eps) + g - 1)
    q = 1.0 / (np.exp(eps) + g - 1)
    noisy_samples = np.zeros([np.size(samples)])

    for i in range(n):
        v = samples[i]
        x = (xxhash.xxh32(str(v), seed=hash_seed[i]).intdigest() % g)
        y = x

        p_sample = np.random.random_sample()
        if p_sample > p - q:
            # perturb
            y = np.random.randint(0, g)
        noisy_samples[i] = y

    # fake value server allocates hash function
    if beta > 0:
        print("attack on OLHB server")
        fake_user_num = int(n * beta)
        compromised_user_index = np.arange(n - fake_user_num, n)
        target_item = np.random.choice(domain - 1, size=r, replace=False)
        fake_value = np.zeros([fake_user_num])

        for i in compromised_user_index:    # set fake values by finding maximum value
            target_value_hash_set = []
            max_num = 0
            for x in target_item:
                target_value_hash_set.append(xxhash.xxh32(str(x), seed=hash_seed[i]).intdigest() % g)
            hash_support_target_num = np.max(np.bincount(target_value_hash_set))
            if hash_support_target_num > max_num:
                max_num = hash_support_target_num
                fake_v = np.argmax(np.bincount(target_value_hash_set))
            fake_value[i - (n - fake_user_num)] = fake_v

        noisy_samples = np.concatenate( (noisy_samples[:-fake_user_num], fake_value), axis = 0 )

    if post_type == 0:
        noisy_result_dist = np.zeros(domain)
        for i in range(n):
            for v in range(domain):
                if noisy_samples[i] == (xxhash.xxh32(str(v), seed=hash_seed[i]).intdigest() % g):
                    noisy_result_dist[v] += 1
        a = 1.0 * g / (p * g - 1)
        b = 1.0 * n / (p * g - 1)
        result = a * noisy_result_dist - b
        return result, noisy_samples, noisy_result_dist
    if post_type == 1:
        return norm_sub(n, noisy_samples, hash_seed, p, g, domain)
    if post_type == 2:
        return norm(n, noisy_samples, hash_seed, p, g, domain)
    if post_type == 3:
        return norm_mul(n, noisy_samples, hash_seed, p, g, domain)
    if post_type == 4:
        return norm_cut(n, noisy_samples, hash_seed, p, g, domain)

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
    data = np.load("../data/normal_numerical.npy")
    # data = np.load("../data/taxi_numerical.npy")
    # data = np.load("../data/retirement_numerical.npy")
    # data = np.random.choice(data, np.size(data) // 10)

    size = np.size(data)
    print("size:", size)

    l = np.min(data)
    h = np.max(data)
    true_mean = np.mean((data - l) / (h - l))
    eps = 0.1
    target_value = np.array([1])
    fake_user_num = int(np.size(data) * 0.01)
    attack_type = 0
    post_type = 1
    bins = 32

    result, size = olh(data, l, h, eps, target_value, fake_user_num, attack_type, post_type, domain_bins=bins)


    from dist_shift_measure import shift_measure
    data_dist, _ = np.histogram(data, range=[l, h], bins=bins)
    print(result)

    distance = shift_measure.shift_distance(data_dist / size, result / size, 512)
    print(calc_mean(result / size, bins, size) - true_mean)
    print(distance)


    import matplotlib.pyplot as plt
    bin_index = np.linspace(0, 1, bins + 1)[1:] - 1 / (2 * bins)
    plt.bar(bin_index, result / size, width=1 / bins)
    plt.show()