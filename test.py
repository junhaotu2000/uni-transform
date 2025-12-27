import numpy as np
import torch
from uni_transform import Transform, Rotation

tf = Transform(
    rotation=torch.tensor([[0.99999999, 0.00000000, 0.00000000], [0.00000000, 1.00000000, 0.00000000], [0.00000000, 0.00000000, 1.00000000]]),
    translation=np.array([0.00000000, 0.00000000, 0.00000000]),
    translation_unit="m",
    extra=None,
)

print(tf.to(backend="numpy").as_rotation_6d())