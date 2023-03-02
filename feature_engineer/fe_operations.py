OPS = {
    "arithmetic": ["add", "subtract", "multiply", "divide"],
    "value_convert": ["abss", 'square', 'inverse', 'log', 'sqrt', 'power3'],
    "special_ops": ["None", "delete", "terminate"]
}


def get_ops(n_features_c, n_features_d):
    c_ops = []
    d_ops = []
    arithmetic = OPS["arithmetic"]
    value_convert_c = OPS["value_convert"] * max(1, n_features_c // 8) + max(1, n_features_c // 4) * OPS["special_ops"]
    special_ops = max(1, (n_features_c + n_features_d) // 12) * OPS["special_ops"]

    if n_features_c == 0:
        c_ops = []
    elif n_features_c == 1:
        c_ops.extend(value_convert_c)
    else:
        for i in range(1, 5):
            op = arithmetic[i - 1]
            for j in range((i - 1) * n_features_c, i * n_features_c):
                c_ops.append(op)
        c_ops.extend(value_convert_c)

    d_ops = ["combine" for _ in range(n_features_c + n_features_d)]
    d_ops.extend(special_ops)
    return c_ops, d_ops
