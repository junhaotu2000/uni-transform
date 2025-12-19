from uni_transform import Transform, transform_interpolate
import torch
import numpy as np

tf1 = Transform.from_rep(torch.randn(1,9), from_rep="euler", extra_dims=4)
tf2 = Transform.from_rep(torch.randn(1,9), from_rep="euler", extra_dims=3)

print(tf1.to_rep("euler"))
print(tf2.to_rep("euler"))