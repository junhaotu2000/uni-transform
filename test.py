import torch
import numpy as np
from uni_transform import Transform, TransformManager


tf_base = Transform.from_rep(
    np.array([1, 2, 3, 0, 0, 0, 1, 0.04]),
    from_rep="quat",
    extra_dims=1,
    translation_unit="m",
)

tf_camera = Transform.from_rep(
    np.array([1, 2, 3, 0, 0, 0, 1, 0.04]),
    from_rep="quat",
    extra_dims=1,
    translation_unit="m",
)

tf_object = Transform.from_rep(
    np.array([1, 2, 3, 0, 0, 0, 1, 0.04]),
    from_rep="quat",
    extra_dims=1,
    translation_unit="m",
)

tm = TransformManager()

tm.add("world", "base", tf_base)
tm.add("base", "camera", tf_camera)
tm.add("camera", "object", tf_object)

print(tm.get("world", "object").as_matrix())
print(tm.get("object", "world").as_matrix())
print(tm.get("world", "object").as_matrix() @ tm.get("object", "world").as_matrix())