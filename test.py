from octic_vits import OcticVisionTransformer
from octic_vits.deit_models import *
from octic_vits.dinov2_models import *
import torch

model = OcticVisionTransformer(
).cuda()
# model = hybrid_deit_huge_patch14(pretrained=True).cuda()
# model = hybrid_dinov2_vit_huge_patch16(pretrained=True).cuda()
img = torch.randn(1, 3, 224, 224).cuda()

out = model(img)