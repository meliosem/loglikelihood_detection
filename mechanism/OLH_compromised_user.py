import numpy as np
import xxhash

def calc_mean(dist, bin, size):
    bin_width = 1 / bin
    sampled_data = np.array([])
    for i in range(len(dist)):
        if dist[i] >= 0:
            sampled_data = np.append(sampled_data, np.random.uniform(i * bin_width, (i + 1) * bin_width, int(size * dist[i])))
        else:
            sampled_data = np.append(sampled_data, -1 * np.random.uniform(i * bin_width, (i + 1) * bin_width, int(size * -dist[i])))
    return np.mean(sampled_data)

# def olh(ori_samples, eps, target_value, fake_user_num, post_type, domain_bins):
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

    hash_num = 100
    g = int(round(np.exp(eps))) + 1
    # g = 16   # test robustness with different g
    p = np.exp(eps) / (np.exp(eps) + g - 1)
    q = 1.0 / (np.exp(eps) + g - 1)
    noisy_samples = np.zeros([np.size(samples)])

    for i in range(n):
        v = samples[i]
        x = (xxhash.xxh32(str(v), seed=i).intdigest() % g)
        y = x

        p_sample = np.random.random_sample()
        if p_sample > p - q:
            # perturb
            y = np.random.randint(0, g)
        noisy_samples[i] = y

    optimal_seed_array = np.array([])  # search hash function and values by supporting the most items in the target set
    target_item = np.random.choice(domain - 1, size=r, replace=False)
    fake_user_num = int(n * beta)
    if beta > 0:
        hash_range = 0
        optimal_seed_array = np.array([])
        optimal_fake_v_array = np.array([])
        for _ in range(fake_user_num):
            optimal_seed = 0
            optimal_fake_v = 0
            max_num = 0
            for hash_seed in np.random.choice(np.arange(1e5), size = hash_num, replace=False).astype("int"):  # search fake value and hash function
                target_value_hash_set = []
                for x in target_item:
                    target_value_hash_set.append(xxhash.xxh32(str(x), seed=hash_seed).intdigest() % g)
                hash_support_target_num = np.max(np.bincount(target_value_hash_set))
                if hash_support_target_num >= max_num:
                    max_num = hash_support_target_num
                    optimal_seed = hash_seed
                    optimal_fake_v = np.argmax(np.bincount(target_value_hash_set))

            optimal_seed_array = np.append(optimal_seed_array, optimal_seed)
            optimal_fake_v_array = np.append(optimal_fake_v_array, optimal_fake_v)
            hash_range += hash_num

        optimal_seed_array = optimal_seed_array.astype("int")
        optimal_fake_v_array = optimal_fake_v_array.astype("int")


        fake_value = np.zeros([fake_user_num]) + optimal_fake_v_array

        noisy_samples = np.concatenate( (noisy_samples[:-fake_user_num], fake_value), axis = 0 )


    if post_type == 0:
        #########################################################   aggregation without attack
        # result = np.zeros(domain_bins)
        # for i in range(n):
        #     for v in range(domain_bins):
        #         if noisy_samples[i] == (xxhash.xxh32(str(v), seed=i).intdigest() % g):
        #             result[v] += 1
        # a = 1.0 * g / (p * g - 1)
        # b = 1.0 * n / (p * g - 1)
        # result = a * result - b
        #########################################################

        #########################################################   aggregation with attack but no sampling
        noisy_result_dist = np.zeros(domain)
        if np.size(optimal_seed_array) != 0:
            for i in range(n):
                if i < n - fake_user_num:
                    for v in range(domain):
                        if noisy_samples[i] == (xxhash.xxh32(str(v), seed=i).intdigest() % g):
                            noisy_result_dist[v] += 1
                else:
                    for v in range(domain):
                        if noisy_samples[i] == (xxhash.xxh32(str(v), seed=optimal_seed_array[i - (n-fake_user_num)]).intdigest() % g):
                            noisy_result_dist[v] += 1
        else:
            for i in range(n):
                for v in range(domain):
                    if noisy_samples[i] == (xxhash.xxh32(str(v), seed=i).intdigest() % g):
                        noisy_result_dist[v] += 1
        a = 1.0 * g / (p * g - 1)
        b = 1.0 * n / (p * g - 1)
        result = a * noisy_result_dist - b
        return result, noisy_samples, noisy_result_dist
        #########################################################


        #########################################################   aggregation with attack and sampling
        # noisy_result_dist = np.zeros(domain)
        # if np.size(optimal_seed_array) != 0:
        #     for i in range(n):
        #         if i < n - fake_user_num:
        #             v_list = []
        #             for v in range(domain):
        #                 if noisy_samples[i] == (xxhash.xxh32(str(v), seed=i).intdigest() % g):
        #                     v_list.append(v)
        #             sampled_v = np.random.choice(v_list, size=1)
        #             noisy_result_dist[sampled_v] += 1
        #         else:
        #             v_list = []
        #             for v in range(domain):
        #                 if noisy_samples[i] == (xxhash.xxh32(str(v), seed=optimal_seed_array[i - (n - fake_user_num)]).intdigest() % g):
        #                     v_list.append(v)
        #             sampled_v = np.random.choice(v_list, size=1)
        #             noisy_result_dist[sampled_v] += 1
        # else:
        #     for i in range(n):
        #         v_list = []
        #         for v in range(domain):
        #             if noisy_samples[i] == (xxhash.xxh32(str(v), seed=i).intdigest() % g):
        #                 v_list.append(v)
        #         sampled_v = np.random.choice(v_list, size=1)
        #         noisy_result_dist[sampled_v] += 1
        # return noisy_samples, noisy_result_dist
        #########################################################


    if post_type == 1:
        return norm_sub(n, noisy_samples, fake_user_num, optimal_seed_array, p, g, domain)
    if post_type == 2:
        return norm(n, noisy_samples, fake_user_num, optimal_seed_array, p, g, domain)
    if post_type == 3:
        return norm_mul(n, noisy_samples, fake_user_num, optimal_seed_array, p, g, domain)
    if post_type == 4:
        return norm_cut(n, noisy_samples, fake_user_num, optimal_seed_array, p, g, domain)



def norm_sub(n, noisy_samples, fake_user_num, optimal_seed_array, p, g, domain_bins):  # this is count, not frequency
    print("norm_sub")
    noisy_result = np.zeros(domain_bins)
    if np.size(optimal_seed_array) != 0:
        for i in range(n):
            if i < n - fake_user_num:
                for v in range(domain_bins):
                    if noisy_samples[i] == (xxhash.xxh32(str(v), seed=i).intdigest() % g):
                        noisy_result[v] += 1
            else:
                for v in range(domain_bins):
                    if noisy_samples[i] == (xxhash.xxh32(str(v), seed=optimal_seed_array[i - (n-fake_user_num)]).intdigest() % g):
                        noisy_result[v] += 1
    else:
        for i in range(n):
            for v in range(domain_bins):
                if noisy_samples[i] == (xxhash.xxh32(str(v), seed=i).intdigest() % g):
                    noisy_result[v] += 1

    a = 1.0 * g / (p * g - 1)
    b = 1.0 * n / (p * g - 1)
    result = a * noisy_result - b

    while (np.fabs(sum(result) - n) > 1) or (result < 0).any(): # Norm-Sub
        result[result < 0] = 0
        total = sum(result)
        mask = result > 0
        diff = (n - total) / sum(mask)
        result[mask] += diff

    return result, noisy_result



def norm(n, noisy_samples, fake_user_num, optimal_seed_array, p, g, domain_bins):   # this is count, not frequency
    print("norm")
    # result = np.zeros(domain_bins)
    # for i in range(n):
    #     for v in range(domain_bins):
    #         if noisy_samples[i] == (xxhash.xxh32(str(v), seed=i).intdigest() % g):
    #             result[v] += 1
    # a = 1.0 * g / (p * g - 1)
    # b = 1.0 * n / (p * g - 1)
    # result = a * result - b

    noisy_result = np.zeros(domain_bins)
    if np.size(optimal_seed_array) != 0:
        for i in range(n):
            if i < n - fake_user_num:
                for v in range(domain_bins):
                    if noisy_samples[i] == (xxhash.xxh32(str(v), seed=i).intdigest() % g):
                        noisy_result[v] += 1
            else:
                for v in range(domain_bins):
                    if noisy_samples[i] == (xxhash.xxh32(str(v), seed=optimal_seed_array[i - (n-fake_user_num)]).intdigest() % g):
                        noisy_result[v] += 1
    else:
        for i in range(n):
            for v in range(domain_bins):
                if noisy_samples[i] == (xxhash.xxh32(str(v), seed=i).intdigest() % g):
                    noisy_result[v] += 1

    a = 1.0 * g / (p * g - 1)
    b = 1.0 * n / (p * g - 1)
    result = a * noisy_result - b


    estimates = np.copy(result)
    total = sum(estimates)
    domain_size = len(result)
    diff = (n - total) / domain_size
    estimates += diff
    return estimates, n

def norm_mul(n, noisy_samples, fake_user_num, optimal_seed_array, p, g, domain_bins):
    print("norm_mul")
    # result = np.zeros(domain_bins)
    # for i in range(n):
    #     for v in range(domain_bins):
    #         if noisy_samples[i] == (xxhash.xxh32(str(v), seed=i).intdigest() % g):
    #             result[v] += 1
    # a = 1.0 * g / (p * g - 1)
    # b = 1.0 * n / (p * g - 1)
    # result = a * result - b

    noisy_result = np.zeros(domain_bins)
    if np.size(optimal_seed_array) != 0:
        for i in range(n):
            if i < n - fake_user_num:
                for v in range(domain_bins):
                    if noisy_samples[i] == (xxhash.xxh32(str(v), seed=i).intdigest() % g):
                        noisy_result[v] += 1
            else:
                for v in range(domain_bins):
                    if noisy_samples[i] == (xxhash.xxh32(str(v), seed=optimal_seed_array[i - (n-fake_user_num)]).intdigest() % g):
                        noisy_result[v] += 1
    else:
        for i in range(n):
            for v in range(domain_bins):
                if noisy_samples[i] == (xxhash.xxh32(str(v), seed=i).intdigest() % g):
                    noisy_result[v] += 1

    a = 1.0 * g / (p * g - 1)
    b = 1.0 * n / (p * g - 1)
    result = a * noisy_result - b


    estimates = np.copy(result)
    estimates[estimates < 0] = 0
    total = sum(estimates)
    return estimates * n / total, n

def norm_cut(n, noisy_samples, fake_user_num, optimal_seed_array, p, g, domain_bins):
    print("norm_cut")
    # result = np.zeros(domain_bins)
    # for i in range(n):
    #     for v in range(domain_bins):
    #         if noisy_samples[i] == (xxhash.xxh32(str(v), seed=i).intdigest() % g):
    #             result[v] += 1
    # a = 1.0 * g / (p * g - 1)
    # b = 1.0 * n / (p * g - 1)
    # result = a * result - b

    noisy_result = np.zeros(domain_bins)
    if np.size(optimal_seed_array) != 0:
        for i in range(n):
            if i < n - fake_user_num:
                for v in range(domain_bins):
                    if noisy_samples[i] == (xxhash.xxh32(str(v), seed=i).intdigest() % g):
                        noisy_result[v] += 1
            else:
                for v in range(domain_bins):
                    if noisy_samples[i] == (xxhash.xxh32(str(v), seed=optimal_seed_array[i - (n-fake_user_num)]).intdigest() % g):
                        noisy_result[v] += 1
    else:
        for i in range(n):
            for v in range(domain_bins):
                if noisy_samples[i] == (xxhash.xxh32(str(v), seed=i).intdigest() % g):
                    noisy_result[v] += 1

    a = 1.0 * g / (p * g - 1)
    b = 1.0 * n / (p * g - 1)
    result = a * noisy_result - b

    
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

    result, noisy_result = olh(data, eps, domain, r, beta, post_type)

    est_dist, _ = np.histogram(result, bins=domain)
    est_dist = est_dist / n
    plt.bar(np.arange(domain), result/n)
    plt.title("est distribution")
    plt.show()


    from dist_shift_measure import shift_measure
    data_dist, _ = np.histogram(data, range=[l, h], bins=bins)
    distance = shift_measure.shift_distance(data_dist / size, result / size, 512)
    print(calc_mean(result / size, bins, size) - true_mean)
    print(distance)

    import matplotlib.pyplot as plt
    bin_index = np.linspace(0, 1, bins + 1)[1:] - 1 / (2 * bins)
    plt.bar(bin_index, result / size, width=1 / bins)
    plt.show()