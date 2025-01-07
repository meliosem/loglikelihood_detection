import numpy as np

# class PM(object):
#     def __init__(self):
#         self.eps = 0
#         self.c = 0
#         self.domain_size = 0
#         self.p = 0
#
#     def init_method(self, eps, domain_size):
#         self.eps = eps
#         self.c = (np.exp(eps/2) + 1)/(np.exp(eps/2) - 1)
#         self.p = (np.exp(eps) - np.exp(eps/2)) / (2 * np.exp(eps/2) + 2)
#         self.domain_size = domain_size
#         print("eps=",self.eps, "C=", self.c)

def compute_l_r(c, t):
    l = (c + 1) / 2 * t - (c - 1) / 2
    r = l + c - 1
    return l, r

def randomize(samples, domain_size):
    sample_size = len(samples)
    l = np.min(samples)
    h = np.max(samples)
    k = (1 - (-1)) / (h - l)
    proj_samples = -1 + k * (samples - l)
    # proj_samples = samples * 2 / domain_size - 1
    ns = np.zeros(sample_size)
    y = np.random.uniform(0, 1, sample_size)
    bar = np.exp(self.eps/2) / (np.exp(self.eps/2) + 1)
    q = self.p / np.exp(self.eps)
    # print("bar:", bar)
    for i, sample in enumerate(proj_samples):
        l, r = self.compute_l_r(sample)
        if y[i] < (l + self.c) * q:
            ns[i] = (y[i] / q - self.c)
        elif y[i] < (self.c - 1) * self.p + (l + self.c) * q:
            ns[i] = ((y[i] - (l + self.c) * q) / self.p + l)
        else:
            ns[i] = ((y[i] - (l + self.c) * q - (self.c - 1) * self.p) / q + r)
    return ns

def estimate_mean(samples, fake_samples, eps):
    np.random.shuffle(samples)
    p = (np.exp(eps) - np.exp(eps / 2)) / (2 * np.exp(eps / 2) + 2)
    q = p / np.exp(eps)
    c = (np.exp(eps / 2) + 1) / (np.exp(eps / 2) - 1)

    sample_size = len(samples)
    l = np.min(samples)
    h = np.max(samples)
    k = (1 - (-1)) / (h - l)
    proj_samples = -1 + k * (samples - l)
    # proj_samples = samples * 2 / domain_size - 1
    ns = np.zeros(sample_size)
    y = np.random.uniform(0, 1, sample_size)
    bar = np.exp(eps / 2) / (np.exp(eps / 2) + 1)
    # print("bar:", bar)
    # for i, sample in enumerate(proj_samples):
    #     l = (c + 1) / 2 * sample - (c - 1) / 2
    #     r = l + c - 1
    #     if y[i] < (l + c) * q:
    #         ns[i] = (y[i] / q - c)
    #     elif y[i] < (c - 1) * p + (l + c) * q:
    #         ns[i] = ((y[i] - (l + c) * q) / p + l)
    #     else:
    #         ns[i] = ((y[i] - (l + c) * q - (c - 1) * p) / q + r)
    # print("ns[0]", ns[1])

    l = (c + 1) / 2 * proj_samples - (c - 1) / 2
    r = l + c - 1
    mark_1 = y < (l + c) * q
    mark_2 = y < (c - 1) * p + (l + c) * q
    mark_2 = mark_2 ^ mark_1
    mark_2[mark_2 == -1] = 0
    mark_3 = np.array([not x for x in (mark_1 | mark_2)])
    ns[mark_1] = (y[mark_1] / q - c)
    ns[mark_2] = ((y[mark_2] - (l[mark_2] + c) * q) / p + l[mark_2])
    ns[mark_3] = ((y[mark_3] - (l[mark_3] + c) * q - (c - 1) * p) / q + r[mark_3])
    # print("ns[0]", ns[1])

    fake_user_num = np.size(fake_samples)
    if fake_user_num > 0:
        ns = np.concatenate( (ns[:-fake_user_num], fake_samples), axis = 0)
    # mean = (np.mean(ns) + 1) / 2 * self.domain_size
    # print(ns)
    mean = np.mean((ns + 1) / k + l)
    return ns, mean

def estimate_var(self, samples, attacker_est_sum, attacker_est_sum_of_square, fake_user, target_mean, target_var):
    s = (np.exp(self.eps / 2) + 1) / (np.exp(self.eps / 2) - 1)
    print("s:", s)
    # l_first = 0
    # h_first = self.domain_size
    # k_first = (1 - (-1)) / (h_first - l_first)
    # l_second = 0
    # h_second = self.domain_size ** 2
    # k_second = (1 - (-1)) / (h_second - l_second)

    n = np.size(samples)

    first_half = samples[:int(len(samples) / 2)]
    second_half = samples[int(len(samples) / 2):]
    second_half = np.square(second_half)

    l_first = np.min(first_half)
    h_first = np.max(first_half)
    k_first = (1 - (-1)) / (h_first - l_first)
    first_half = -1 + k_first * (first_half - l_first)
    l_second = np.min(np.square(second_half))
    h_second = np.max(np.square(second_half))
    k_second = (1 - (-1)) / (h_second - l_second)
    second_half = -1 + k_second * (second_half - l_second)

    ns_first = self.randomize(first_half, self.domain_size)

    fake_value_sum = target_mean * (n + fake_user) / 2 - attacker_est_sum / 2
    fake_value_sum_output = (fake_value_sum - fake_user / 2 * l_first) * k_first - fake_user / 2
    fake_value_for_mean = np.array([fake_value_sum_output / (fake_user // 2)] * (fake_user // 2))

    ns_first = np.concatenate((ns_first, fake_value_for_mean), axis = 0)

    mean = (np.mean(ns_first) + 1) / k_first + l_first
    print("Poisoned mean:", mean)

    ns_second = self.randomize(second_half, self.domain_size ** 2)

    fake_value_sum_of_square = (target_var + target_mean ** 2) * (n + fake_user)/2 - attacker_est_sum_of_square / 2
    fake_value_sum_of_square_output = (fake_value_sum_of_square - fake_user / 2 * l_second) * k_second - fake_user / 2
    fake_value_for_var = np.array([fake_value_sum_of_square_output / (fake_user // 2)] * (fake_user // 2))

    ns_second = np.concatenate((ns_second, fake_value_for_var), axis = 0)

    var = np.mean( (ns_second + 1) / k_second + l_second ) - mean ** 2

    return mean, var