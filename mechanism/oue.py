import numpy as np

def oue_perturb(ori_samples, eps, domain):
    p = 1/2
    q = 1 / (np.exp(eps) + 1)

    np.random.shuffle(ori_samples)
    n = np.size(ori_samples)
    noisy_samples = np.zeros([np.size(ori_samples), domain])

    # encode by loop (slow)
    # noisy_samples = np.zeros([np.size(samples), domain])
    # bin_width = 1 / domain
    # for i in range(np.size(samples)):
    #     index = int(samples[i] / bin_width)
    #     if index == domain:
    #         index -= 1
    #     noisy_samples[i, index] = 1

    # encode by matrix (fast)
    index = ori_samples.astype("int")
    index = list(index)
    noisy_samples = np.eye(domain)[index]

    # report by loop (slow)
    # for i in range(np.size(samples)):
    #     index = int(samples[i] / bin_width)
    #     if index == domain:
    #         index -= 1
    #     randoms = np.random.uniform(0, 1, domain)
    #     if randoms[index] > p:
    #         noisy_samples[i, index] = 0
    #     mark = randoms < q
    #     mark[index] = 0
    #     noisy_samples[i, mark] = 1

    # report by matrix (fast)
    randoms = np.random.uniform(0, 1, [np.size(ori_samples), domain])
    mark_1 = noisy_samples == 1
    mark_0 = noisy_samples == 0
    mark_1_flip = (randoms > p) & mark_1
    noisy_samples[mark_1_flip] = 0
    mark_0_flip = (randoms < q) & mark_0
    noisy_samples[mark_0_flip] = 1

    return noisy_samples

def oue_aggregator(noisy_samples, eps, domain):
    n = int(noisy_samples.shape[0])
    # n = int(np.size(noisy_samples))
    p = 1 / 2
    q = 1 / (np.exp(eps) + 1)
    result = np.sum(noisy_samples, axis=0)
    result = (result - n * q) / (p - q)
    return result