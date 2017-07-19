

import numpy as np
import argparse

from simulate import *

sim = Simulator()


parser = argparse.ArgumentParser(description='Simulates particles in 2D')
parser.add_argument('--seed', type=int, default=0, required=False)
parser.add_argument('--N', type=int, default=10, required=False)
parser.add_argument('--M', type=int, default=20, required=False)

args = parser.parse_args()
seed = args.seed
N = args.N
nperevent = args.M

Mmin = 1
Mmax = 1

data = pd.DataFrame({'event': [0], 'particle': [0], 'hit': [
                    0], 'layer': [0], 'iphi': [0], 'x': [0.], 'y': [0.]})
data = data.drop(data.index[[0]])

data_particle = pd.DataFrame({'event': [0], 'particle': [0], 'pt': [
                             0.], 'phi': [0.], 'xVtx': [0.], 'yVtx': [0.]})
data_particle = data_particle.drop(data_particle.index[[0]])

np.random.seed(seed)
event_offset = seed * N

print("Will now produce ", N, " events with in average", nperevent, " tracks")
for ievent in range(event_offset, N + event_offset):

    if(ievent % 1 == 0):
        print("processing event : ", ievent)
    event = pd.DataFrame({
        'event': [0], 'particle': [0], 'hit': [0],
        'layer': [0], 'iphi': [0],
        'x': [0.], 'y': [0.]})
    event = event.drop(event.index[[0]])

    event_particle = pd.DataFrame({
        'event': [0], 'particle': [0],
        'pt': [0.], 'phi': [0.], 'xVtx': [0.], 'yVtx': [0.]})
    event_particle = event_particle.drop(event_particle.index[[0]])

    M = 0
    while M == 0:
        M = np.random.poisson(nperevent)

    sim.detector.reset()

    for p in range(0, M):  # generate M tracks
        d = 2. / 3  # d0 spread as in atlas hl lhc study
        v = 3.
        position = np.array([np.random.normal(0., d), np.random.normal(0., d)])
        pt = np.random.uniform(300, 10000)
        phi = np.random.uniform(-np.pi, np.pi)
        momentum = np.array([pt * np.cos(phi), pt * np.sin(phi)])
        xVtx = position[0]
        yVtx = position[1]
        charge = 2 * np.random.random_integers(0, 1) - 1

        # DR simtrack=sim.propagate(position,velocity, step = 20, id=p)
        simtrack = sim.propagate(position, momentum, charge=charge, id=p)
        simtrack = pd.concat(
            [pd.DataFrame({'event': [ievent] * len(simtrack.index)}),
             simtrack],
            axis=1
        )
        event = event.append(simtrack, ignore_index=True)

        event_particle = event_particle.append(pd.concat(
            [pd.DataFrame({'event': [ievent], 'particle':[p],
                           'pt':[charge * pt], 'phi':[phi],
                           'xVtx':[xVtx], 'yVtx':[yVtx]})])
        )

    hits = sim.detector.getHits()
    hits = hits.iloc[np.random.permutation(len(hits.index))]
    hits = hits.reset_index(drop=True)
    data_event = pd.concat([
        pd.DataFrame({'event': [ievent] * len(hits.index)}),
        hits],
        axis=1
    )

    data = data.append(data_event, ignore_index=True)
    data_particle = data_particle.append(event_particle, ignore_index=True)

for col in ['event', 'particle', 'hit', 'layer', 'iphi']:
    data[col] = data[col].astype('int32')

for col in ['event', 'particle']:
    data_particle[col] = data_particle[col].astype('int32')

data = data.drop(['hit'], axis=1)

# precision could probably be reduced
data.to_csv("hits_" + str(N) + "_" + str(seed) + ".csv", header=(seed == 0),
            columns=['event', 'particle', 'layer', 'iphi', 'x', 'y'],
            index=False)
data_particle.to_csv(
    "particles_" + str(N) + "_" +
    str(seed) + ".csv", header=(seed == 0),
    columns=['event', 'particle', 'pt', 'phi', 'xVtx', 'yVtx'],
    index=False)
