import torch

# output = output.view(batch * num_anchors, 5 + num_classes, h * w).transpose(0, 1).contiguous().view(5 + num_classes,
#                                                                                                     batch * num_anchors * h * w)

tnsr = torch.Tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]])

print(tnsr.view(1, 3, 4).shape)
print(tnsr.view(1, 3, 4).contiguous().transpose(0, 2).view(4,3))