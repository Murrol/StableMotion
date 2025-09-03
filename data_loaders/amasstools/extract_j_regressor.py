import numpy as np
from data_loaders.amasstools.joints import JOINTS_EXTRACTOR


data = np.load("data_loaders/amasstools/smpl_neutral_nobetas_J.npz")
J = data["J"]
parents = data["parents"]
smplh2smpl = JOINTS_EXTRACTOR["smpljoints"]

new_parents = data["parents"][smplh2smpl]
new_J = J[smplh2smpl]

np.savez("data_loaders/amasstools/smpl_neutral_nobetas_24J.npz", J=new_J, parents=new_parents)