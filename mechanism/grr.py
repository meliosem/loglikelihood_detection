import numpy as np



def targeted_attack(target, beta, x, eps, domain, n):
	np.random.shuffle(x)
	y = np.copy(x)
	p = np.exp(eps) / (np.exp(eps) + domain - 1)
	q = 1 / (np.exp(eps) + domain - 1)

	randoms = np.random.random(size = np.size(y))
	mark_perturb = randoms > p - q
	num_perturb = np.sum(mark_perturb)
	perturb_result = np.random.randint(domain, size = num_perturb)
	y[mark_perturb] = perturb_result


	fake_user_num = int(n * beta)
	fake_value = np.zeros(fake_user_num)
	fake_value += target

	y = np.concatenate( (y[:-fake_user_num], fake_value), axis = 0 )
	return y



def grr_perturbation(x, eps, domain):
	y = np.copy(x)
	p = np.exp(eps) / (np.exp(eps) + domain - 1)
	q = 1 / (np.exp(eps) + domain - 1)
	randoms = np.random.random(size = np.size(y))
	mark_perturb = randoms > p - q
	num_perturb = np.sum(mark_perturb)
	perturb_result = np.random.randint(0, domain, size = num_perturb)
	y[mark_perturb] = perturb_result
	return y

def grr_aggregation(y, eps, domain, n):
	p = np.exp(eps) / (np.exp(eps) + domain - 1)
	q = 1 / (np.exp(eps) + domain - 1)
	result = np.zeros(domain)
	for i in range(domain):
		mark = y == i
		noisy_sum = np.sum(mark)
		result[i] = (noisy_sum - n * q) / (p - q)
	return result


if __name__ == "__main__":
	# data = np.random.randint(100, size = 100000)
	data = np.random.normal(0, 10, size=100000)
	h = np.max(data)
	l = np.min(data)
	data = (data - l) / (h - l)
	domain_bins = 32
	bin_width = 1 / domain_bins
	index = data / bin_width
	index = index.astype("int")
	mark = index == domain_bins
	index[mark] -= 1
	data = list(index)
	eps = 1

	noisy_result = grr_perturbation(data, eps, domain_bins)
	result = grr_aggregation(noisy_result, eps, domain_bins, 100000)

	import matplotlib.pyplot as plt
	fig = plt.figure()

	plt.bar(np.arange(domain_bins), result)
	plt.show()