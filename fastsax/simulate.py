import json
import os
import time
from collections import defaultdict

import pandas as pd

import fastsax
from fastsax.main import operations


# ### Physics


def construct_configuration():
    # In[70]:
    nr = json.load(open(os.path.join(fastsax.helpers.SAX_DIR, 'config', 'ER.json')))
    c = {key: value['initial'] for key, value in nr.items()}

    c['py'] = [c['py%d' % i] for i in range(5)]

    # In[71]:

    c['reference'] = {}
    # This is the reference curve for recombination fluctuation
    c['reference']['rf'] = {'k': (0.1882837602,
                                  0.0346653842, -0.0141909098,
                                  0.0018700269, -0.0001192586, 0.0000036759,
                                  -0.0000000429
                                  ),
                            }

    # This is the reference curve for photon yield
    c['reference']['py'] = {'k': (
        -1.2082551807,  # -10 + 8.791744819312653,
        10.629041834795597,
        -1.241413827588842,
        0.079120757139136,
        -0.002173210112388,
        -0.000005992712788,
        0.000001177434501,
        0.000000005044714,
        -0.000000000817828,
        0.000000000009275
    ),
        'threshold': 0.5,  # Energy below which fully recombines
    }

    c['reference']['rf']['range'] = (0, 1)

    c['setup'] = dict(n=int(1e6),
                      E=(0, 40),  # keV
                      z=(-92.9, -9),  # cm
                      r=(0, 36.94),  # cm
                      electron_lifetime=600,  # us
                      )

    c['recoil_type'] = 'ER'

    return c


df = pd.DataFrame()


# get_ipython().magic('timeit electron_detection(dpe_probability(photon_detection(electron_lifetime_prob(s1_s2_inversed_correction_factor(photon_electron_number(recombination_fluctuation(excimer_ion_distributing(heat_quenching(quanta(recombination(initial_conditions(df))))))))))))')

# %prun -l 4 electron_detection(dpe_probability(photon_detection(electron_lifetime_prob(s1_s2_inversed_correction_factor(photon_electron_number(recombination_fluctuation(excimer_ion_distributing(heat_quenching(quanta(recombination(initial_conditions(df))))))))))))


# In[25]:

# import cProfile, pstats, io

# df = pd.DataFrame()
# pr = cProfile.Profile()
# pr.enable()
# df = electron_detection(dpe_probability(photon_detection(electron_lifetime_prob(s1_s2_inversed_correction_factor(photon_electron_number(recombination_fluctuation(excimer_ion_distributing(heat_quenching(quanta(recombination(initial_conditions(df))))))))))))
# pr.disable()
# pr.dump_stats('profile.prof')



# In[38]:
def run(operations=operations,
        c=construct_configuration(),
        stats=False,
        n=1):
    df_initial = operations[0](pd.DataFrame(), c)

    times = defaultdict(list)

    for i in range(n):
        df = df_initial.copy()

        for operation in operations[1:]:
            print(operation.__name__)
            t0 = time.time()
            df = operation(df, c)
            t1 = time.time()

            times[operation.__name__].append(t1 - t0)

    if not stats:
        return df

    df_stats = pd.DataFrame(times)
    df_stats['speed'] = pd.DataFrame(times).sum(axis=1)

    return df, df_stats


    # In[ ]:


if __name__ == '__main__':
    print(run(n=10, stats=True)[1].describe())
