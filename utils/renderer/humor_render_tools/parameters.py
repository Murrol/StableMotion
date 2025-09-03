# adapted from HuMoR
# fmt: off
smpl_connections = [[11, 8], [8, 5], [5, 2], [2, 0], [10, 7], [7, 4], [4, 1], [1, 0],
                    [0, 3], [3, 6], [6, 9], [9, 12], [12, 15], [12, 13], [13, 16], [16, 18],
                    [18, 20], [12, 14], [14, 17], [17, 19], [19, 21]]

# VERTEX_COLOR = [0.0390625, 0.4140625, 0.796875]
VERTEX_COLOR = [0.631, 0.788, 0.957]
# VERTEX_COLOR = (0.725, 0.949, 0.941)
# GENVERTEX_COLOR = [1.0, 0.714, 0.757]
GENVERTEX_COLOR = (1.0, 0.714, 0.725)

colors = {
    'pink': [.7, .7, .9],
    'purple': [.9, .7, .7],
    'cyan': [.7, .75, .5],
    'red': [1.0, 0.0, 0.0],
    'green': [.0, 1., .0],
    'yellow': [1., 1., 0],
    'brown': [.5, .7, .7],
    'blue': [.0, .0, 1.],
    'offwhite': [.8, .9, .9],
    'orange': [.5, .65, .9],
    'grey': [.7, .7, .7],
    'black': [0.0, 0.0, 0.0],
    'white': [1.0, 1.0, 1.0],
    'yellowg': [0.83, 1.0, 0.0],
    "vertex": VERTEX_COLOR,
    "genvertex": GENVERTEX_COLOR,
}
# fmt: on
