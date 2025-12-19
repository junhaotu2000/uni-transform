from uni_transform import Transform, transform_interpolate
import torch
import numpy as np

tf0 = Transform.identity(backend="torch")
tf1 = Transform.from_rep(torch.randn(1000,200,32,10), from_rep="euler", extra_dims=1)

t = tf0.relative_to(tf1)
print(t.to_rep("euler").shape)