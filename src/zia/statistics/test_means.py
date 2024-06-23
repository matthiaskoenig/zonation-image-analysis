import numpy as np

stds_data = [40.6,
             28.5,
             42.0,
             38.8,
             33.9,
             56.1

             ]

means_data = [
    75.6,
    69.9,
    77.9,
    76.7,
    72.6,
    92.5,

]

ns_data = [322,
           332,
           263,
           239,
           267,
           157

           ]


def calc_mean_var(ns, stds):
    return np.sum([n * s ** 2 for n, s in zip(ns, stds)]) / np.sum(ns)


def calc_mean_means(ns, means):
    return np.sum([n * mean for n, mean in zip(ns, means)]) / np.sum(ns)


def calc_var_mean(ns, means):
    m_o_m = calc_mean_means(ns, means)
    return np.sum([n * (mean - m_o_m) ** 2 for n, mean in zip(ns, means)]) / np.sum(ns)


print(np.sqrt(calc_var_mean(ns_data, means_data) + calc_mean_var(ns_data, stds_data)))
