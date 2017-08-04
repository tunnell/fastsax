import inspect
import os

import numpy as np

SAX_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


def initial_conditions(df, c):
    """Step 0: Setup simulation conditions

    This will create an energy and spatial distribution of events to simulate.
    """
    # How many events to simulate
    n = c['setup']['n']

    # Energy
    df['E'] = np.random.uniform(*c['setup']['E'],
                                size=n)

    # Depth
    df['z'] = np.random.uniform(*c['setup']['z'],
                                size=n)

    # Radius squared
    df['r2'] = np.random.uniform(np.power(c['setup']['r'][0],
                                          2.),
                                 np.power(c['setup']['r'][1],
                                          2.),
                                 size=n)

    # Angle
    df['theta'] = np.random.uniform(0, 2. * np.pi,
                                    size=n)

    # x, y position of event
    df['x'] = np.sqrt(df['r2']) * np.cos(df['theta'])
    df['y'] = np.sqrt(df['r2']) * np.sin(df['theta'])

    # Clear temporary variables
    del df['theta']
    del df['r2']

    return df


class SaxUnphysicalModel(Exception):
    pass
