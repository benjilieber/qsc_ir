import math

import matplotlib.pyplot as plt

from polar.polar_cfg import PolarCfg


q=3
p_err = 0.0
N=1024
constr_l=100
c = PolarCfg(q=q,
             p_err=p_err,
             N=N,
             num_info_indices=0,
             constr_l=constr_l,
             use_log=True,
             verbosity=True)
num_info_indices_list, success_prob_list = c.get_num_info_indices_to_success_prob_map()
rate_list = [num_info_indices * math.log2(c.q) / c.N for num_info_indices in num_info_indices_list]

plt.plot(rate_list, success_prob_list)
plt.axvline(c._theoretic_key_rate())

print([(rate, success_prob) for rate, success_prob in zip(rate_list, success_prob_list)])

plt.xlabel('rate')
plt.ylabel('success_rate')

plt.show()