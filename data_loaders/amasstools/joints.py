import numpy as np


JOINT_NAMES = {
    "smpljoints": [
        "pelvis",
        "left_hip",
        "right_hip",
        "spine1",
        "left_knee",
        "right_knee",
        "spine2",
        "left_ankle",
        "right_ankle",
        "spine3",
        "left_foot",
        "right_foot",
        "neck",
        "left_collar",
        "right_collar",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hand",
        "right_hand",
    ],
}

JOINT_NAMES_IDX = {
    jointstype: {x: i for i, x in enumerate(JOINT_NAMES[jointstype])}
    for jointstype in JOINT_NAMES
}


# EXTRACTOR from SMPLH layer
# replace the "left_hand", "right_hand" by "left_index1", "right_index1" of SMPLH
JOINTS_EXTRACTOR = {
    "smpljoints": np.array(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            37,
        ]
    )
}

# [1, 4, 7, 10, 13, 16, 18, 20, 22] for smpljoints
LEFT_CHAIN = {
    jointstype: np.array(
        [i for x, i in JOINT_NAMES_IDX[jointstype].items() if "left" in x]
    )
    for jointstype in JOINT_NAMES
}

# [2, 5, 8, 11, 14, 17, 19, 21, 23] for smpljoints
RIGHT_CHAIN = {
    jointstype: np.array(
        [i for x, i in JOINT_NAMES_IDX[jointstype].items() if "right" in x]
    )
    for jointstype in JOINT_NAMES
}


INFOS = {
    "smpljoints": {
        "LM": JOINT_NAMES["smpljoints"].index("left_ankle"),
        "RM": JOINT_NAMES["smpljoints"].index("right_ankle"),
        "LF": JOINT_NAMES["smpljoints"].index("left_foot"),
        "RF": JOINT_NAMES["smpljoints"].index("right_foot"),
        "LS": JOINT_NAMES["smpljoints"].index("left_shoulder"),
        "RS": JOINT_NAMES["smpljoints"].index("right_shoulder"),
        "LH": JOINT_NAMES["smpljoints"].index("left_hip"),
        "RH": JOINT_NAMES["smpljoints"].index("right_hip"),
        "njoints": len(JOINT_NAMES["smpljoints"]),
    }
}
