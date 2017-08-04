#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `fastsax` package."""

import json
import os


def test_fail():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')
    n = int(1e6)

    # In[69]:

    E = (0, 10)  # keV
    z = (-92.9, -9)  # cm
    r = (0, 36.94)  # cm
    electron_lifetime = 600  # us


def test_load():
    """Sample pytest test function with the pytest fixture as an argument."""


def test_run():
    """Sample pytest test function with the pytest fixture as an argument."""
    from .helpers import SAX_DIR
    n = int(1e6)

    # In[69]:

    E = (0, 10)  # keV
    z = (-92.9, -9)  # cm
    r = (0, 36.94)  # cm
    electron_lifetime = 600  # us

    # ### Physics

    # In[70]:
    nr = json.load(open(os.path.join(SAX_DIR, 'config', 'NR.json')))

    # In[71]:

    reference = {}
    reference['rf'] = {'k': (0.1882837602,
                             0.0346653842, -0.0141909098,
                             0.0018700269, -0.0001192586, 0.0000036759,
                             -0.0000000429
                             ),
                       }

    reference['py'] = {'k': (
        4.923538039112815,
        13.811307382341761,
        -2.460241551261530,
        0.282503542158447,
        -0.018797635055821,
        0.000655006486882,
        -0.000009212424211
    ),
    }

    # In[72]:

    c = {key: value['initial'] for key, value in nr.items()}

    # In[73]:

    reference['rf']['range'] = (0, 1)
    reference['py']['range'] = [c['alpha'] / (1. + c['alpha']) / c['W'],
                                1. / c['W']]
