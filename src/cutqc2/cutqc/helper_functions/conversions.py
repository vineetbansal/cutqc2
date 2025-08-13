import numpy as np


def reverseBits(num, bitSize):
    binary = bin(num)
    reverse = binary[-1:1:-1]
    reverse = reverse + (bitSize - len(reverse)) * "0"
    return int(reverse, 2)


def reverse_prob(prob_l):
    nqubit = int(np.log(len(prob_l)) / np.log(2))
    assert 2**nqubit == len(prob_l)
    reverse_prob_l = np.zeros(2**nqubit, dtype=float)
    for state, p in enumerate(prob_l):
        reverse_state = reverseBits(num=state, bitSize=nqubit)
        reverse_prob_l[reverse_state] = p
    return reverse_prob_l


def list_to_dict(l):
    l_dict = {}
    num_qubits = int(np.log(len(l)) / np.log(2))
    assert 2**num_qubits == len(l)
    for state, entry in enumerate(l):
        bin_state = bin(state)[2:].zfill(num_qubits)
        l_dict[bin_state] = entry
    if abs(sum(l_dict.values()) - sum(l)) > 1:
        print(
            "list_to_dict may be wrong, converted counts = {}, input counts = {}".format(
                sum(l_dict.values()), sum(l)
            )
        )
    return l_dict


def dict_to_array(distribution_dict, force_prob):
    state = list(distribution_dict.keys())[0]
    num_qubits = len(state)
    num_shots = sum(distribution_dict.values())
    cnts = np.zeros(2**num_qubits, dtype=float)
    for state in distribution_dict:
        cnts[int(state, 2)] = distribution_dict[state]
    if abs(sum(cnts) - num_shots) > 1:
        print(
            "dict_to_array may be wrong, converted counts = {}, input counts = {}".format(
                sum(cnts), num_shots
            )
        )
    if not force_prob:
        return cnts
    else:
        prob = cnts / num_shots
        assert abs(sum(prob) - 1) < 1e-10
        return prob


def memory_to_dict(memory):
    mem_dict = {}
    for m in memory:
        if m in mem_dict:
            mem_dict[m] += 1
        else:
            mem_dict[m] = 1
    assert sum(mem_dict.values()) == len(memory)
    return mem_dict


def quasi_to_real(quasiprobability, mode):
    """
    Convert a quasi probability to a valid probability distribution
    """
    if mode == "nearest":
        return nearest_probability_distribution(quasiprobability=quasiprobability)
    elif mode == "naive":
        return naive_probability_distribution(quasiprobability=quasiprobability)
    else:
        raise NotImplementedError("%s conversion is not implemented" % mode)


def nearest_probability_distribution(quasiprobability):
    """Takes a quasiprobability distribution and maps
    it to the closest probability distribution as defined by
    the L2-norm.
    Parameters:
        return_distance (bool): Return the L2 distance between distributions.
    Returns:
        ProbDistribution: Nearest probability distribution.
        float: Euclidean (L2) distance of distributions.
    Notes:
        Method from Smolin et al., Phys. Rev. Lett. 108, 070502 (2012).
        Vectorized implementation for improved performance and numerical stability.
        
        This vectorized approach solves the optimization problem:
        minimize ||p - q||_2^2 subject to p >= 0 and sum(p) = 1
        
        The solution has the form p_i = max(q_i - theta, 0) where theta
        is chosen to satisfy the normalization constraint.
    """
    q = np.asarray(quasiprobability)
    n = len(q)

    # Sort in descending order to efficiently find the threshold
    u = np.sort(q)[::-1]

    # Cumulative sum of sorted values
    cssv = np.cumsum(u)
    
    # Find the largest index rho such that u[rho] - theta > 0
    # where theta = (sum(u[0:rho+1]) - 1) / (rho + 1)
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0][-1]
    
    # Compute the threshold value
    theta = (cssv[rho] - 1) / (rho + 1)
    
    # Apply threshold and clip to ensure non-negativity
    p = np.maximum(q - theta, 0)

    return p


def naive_probability_distribution(quasiprobability):
    """
    Takes a quasiprobability distribution and does the following two steps:
    1. Update all negative probabilities to 0
    2. Normalize
    """
    new_probs = np.where(quasiprobability < 0, 0, quasiprobability)
    new_probs /= np.sum(new_probs)
    return new_probs
