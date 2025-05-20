import torch
import torch.nn.functional as F

from octic_vits.d8_utils import (
    group_elements,
    irreps,
    mult_table,
    isotypic_group_action, 
    regular_group_action,
    image_space_group_action,
    spatial_and_isotypic_group_action,
    isotypic_to_regular_D8,
    regular_to_isotypic_D8,
    convert_5tuple_to_8tuple, convert_8tuple_to_5tuple
)


from octic_vits.d8_invariantization import (
    LinearInvariant,
    PowerSpectrumInvariant,
    PolynomialInvariant,
    MaxFilteringInvariant,
    ThirdOrderInvariant,
    CanonizationInvariant,
)

from octic_vits.d8_layers import (
    TritonGeluD8,
    LinearD8,
    LayerNormD8,
    MlpD8,
    LiftD8,
    PatchEmbedD8,
    IsotypicToPatchD8,
    AttentionD8,
    BlockD8,
)

from octic_vits.model import OcticVisionTransformer

# from  d8_components.d8_blocks import (
#     EncoderBlockD8,
# )

# from mae.d8_mae import MaskedAutoencoderViTD8
# from d8_components.d8_blocks import BlockD8
from tqdm import tqdm
# from deit.d8_vit import d8_vit_models
# from deit.d8_vit_inv_early import d8_inv_early_vit_models

def test_group_action():
    xs = tuple(torch.randn(2, 64, 256) for _ in range(8))
    a = isotypic_group_action
    for mult in mult_table:
        g1, g2, g1g2 = mult
        for x, y, irrep in zip(a(g1, a(g2, xs)), a(g1g2, xs), irreps):
            assert torch.allclose(x, y), \
                f"incorrect {irrep} multiplication {g1}, {g2} -> {g1g2}"
    print("Isotypic group action test passed!")
    a = regular_group_action
    for mult in mult_table:
        g1, g2, g1g2 = mult
        for x, y in zip(a(g1, a(g2, xs)), a(g1g2, xs)):
            assert torch.allclose(x, y), \
                f"incorrect regular rep multiplication {g1}, {g2} -> {g1g2}"
    print("Regular group action test passed!")

def test_spatial_and_isotypic_group_action(device="cuda"):
    xs = tuple(torch.randn(32, 196, 48).to(device) for _ in range(8))
    a = spatial_and_isotypic_group_action
    for mult in mult_table:
        g1, g2, g1g2 = mult
        for x, y in zip(a(g1, a(g2, xs)), a(g1g2, xs)):
            assert torch.allclose(x, y), \
                f"incorrect spatial+isotypic multiplication {g1}, {g2} -> {g1g2}"
    print("Spatial and isotypic group action test passed!")

def test_image_space_group_action(device="cuda"):
    img = torch.randn(32, 3, 224, 224).to(device)
    a = image_space_group_action
    for mult in mult_table:
        g1, g2, g1g2 = mult
        assert torch.allclose(a(g1, a(g2, img)), a(g1g2, img)), \
            f"incorrect image space multiplication {g1}, {g2} -> {g1g2}"
    print("Image space group action test passed!")

def test_fourier_transforms_inverses():
    xs = tuple(torch.randn(2, 64, 256) for _ in range(8))
    for x, y in zip(
        regular_to_isotypic_D8(isotypic_to_regular_D8(xs)),
        xs,
    ):
        assert torch.allclose(x, y, atol=1e-6), \
            "reg_to_iso(iso_to_reg(x)) is not identity"
    for x, y in zip(
        isotypic_to_regular_D8(regular_to_isotypic_D8(xs)),
        xs,
    ):
        assert torch.allclose(x, y, atol=1e-6), \
            "iso_to_reg(reg_to_iso(x)) is not identity"
    print("Fourier transform are each others inverses tested")

def test_fourier_transforms():
    xs = tuple(torch.randn(2, 64, 256) for _ in range(8))
    for g in group_elements:
        for x, y in zip(
            isotypic_group_action(g, regular_to_isotypic_D8(xs)),
            regular_to_isotypic_D8(regular_group_action(g, xs))
        ):
            assert torch.allclose(x, y, atol=1e-6), \
                f"group actions in isotypic and regular space don't match for {g}"
    print("Passed test of transform to isotypic")
    for g in group_elements:
        for x, y in zip(
            isotypic_to_regular_D8(isotypic_group_action(g, xs)),
            regular_group_action(g, isotypic_to_regular_D8(xs))
        ):
            assert torch.allclose(x, y, atol=1e-6), \
                f"group actions in isotypic and regular space don't match for {g}"
    print("Passed test of transform to regular")

def test_equi_two_isotypic_to_isotypic(layer, layer_name, irrep_size=[128, 64, 256], device="cuda"):
    with torch.inference_mode():
        xs = tuple(
            torch.randn(irrep_size, device=device) + torch.randn(*irrep_size[:-1], 1, device=device) 
            for _ in range(8)
        )
        ys = tuple(
            torch.randn(irrep_size, device=device) + torch.randn(*irrep_size[:-1], 1, device=device) 
            for _ in range(8)
        )
        for g in group_elements:
            for x1, x2, irrep in zip(
                isotypic_group_action(g, layer(xs, ys)),
                layer(isotypic_group_action(g, xs), isotypic_group_action(g, ys)),
                irreps
            ):
                assert not torch.allclose(x1, torch.zeros_like(x1), atol=1e-6), \
                    f"Bad test: {layer_name} outputs 0 with irrep {irrep} and group element {g}."
                assert torch.allclose(x1, x2, atol=1e-6), \
                    f"{layer_name} doesn't commute with group action, irrep: {irrep}, group element: {g}."
    print(f"Passed test of {layer_name}")


def test_equi_isotypic_to_isotypic(layer, layer_name, irrep_size=[128, 64, 256], device="cuda"):
    with torch.inference_mode():
        xs = tuple(
            torch.randn(irrep_size, device=device) + torch.randn(*irrep_size[:-1], 1, device=device) 
            for _ in range(8)
        )
        for g in group_elements:
            for x, y, irrep in zip(
                isotypic_group_action(g, convert_5tuple_to_8tuple(layer(convert_8tuple_to_5tuple(xs)))),
                convert_5tuple_to_8tuple(layer(convert_8tuple_to_5tuple(isotypic_group_action(g, xs)))),
                irreps
            ):
                assert not torch.allclose(x, torch.zeros_like(x), atol=1e-6), \
                    f"Bad test: {layer_name} outputs 0 with irrep {irrep} and group element {g}."
                assert torch.allclose(x, y, atol=1e-6), \
                    f"{layer_name} doesn't commute with group action, irrep: {irrep}, group element: {g}."
    print(f"Passed test of {layer_name}")

def test_equi_gelu_d8():
    gelu8 = TritonGeluD8()
    test_equi_isotypic_to_isotypic(gelu8, "GeluD8")

def test_equi_linear_d8():
    device = "cuda"
    lin8 = LinearD8(8*256, 384).to(device)
    test_equi_isotypic_to_isotypic(lin8, "LinearD8", device=device)

def test_equi_layernorm_d8():
    device = "cuda"
    ln8 = LayerNormD8(8*256).to(device)
    test_equi_isotypic_to_isotypic(ln8, "LayerNormD8v2", device=device)

def test_equi_mlp_d8():
    device = "cuda"
    f8 = MlpD8(8*256).to(device)
    test_equi_isotypic_to_isotypic(f8, "MlpD8", device=device)

def test_equi_img_to_isotypic(layer, layer_name, device="cuda"):
    with torch.inference_mode():
        img = torch.randn(32, 3, 224, 224, device=device)
        for g in group_elements:
            for x, y, irrep in zip(
                (image_space_group_action(g, feat) for feat in isotypic_group_action(g, layer(img))),
                layer(image_space_group_action(g, img)),
                irreps
            ):
                assert not torch.allclose(x, torch.zeros_like(x), atol=1e-5), \
                    f"Bad test: {layer_name} outputs 0 with irrep {irrep} and group element {g}."
                assert torch.allclose(x, y, atol=1e-5), \
                    f"{layer_name} doesn't commute with group action, irrep: {irrep}, group element: {g}."
    print(f"Passed test of {layer_name}")

def test_equi_img_to_flattened_isotypic(layer, layer_name, device="cuda"):
    with torch.inference_mode():
        img = torch.randn(32, 3, 224, 224, device=device)
        for g in group_elements:
            for x, y, irrep in zip(
                spatial_and_isotypic_group_action(g, convert_5tuple_to_8tuple(layer(img))),
                convert_5tuple_to_8tuple(layer(image_space_group_action(g, img))),
                irreps
            ):
                assert not torch.allclose(x, torch.zeros_like(x), atol=1e-5), \
                    f"Bad test: {layer_name} outputs 0 with irrep {irrep} and group element {g}."
                assert torch.allclose(x, y, atol=1e-5), \
                    f"{layer_name} doesn't commute with group action, irrep: {irrep}, group element: {g}."
    print(f"Passed test of {layer_name}")

def test_equi_flattened_isotypic_to_img(layer, layer_name, dim=768, device="cuda"):
    if dim % 8 != 0: raise ValueError()
    with torch.inference_mode():
        xs = tuple(torch.randn(32, 196, dim//8, device=device) for _ in range(8))
        for g in group_elements:
            x = image_space_group_action(g, layer(xs)),
            y = layer(spatial_and_isotypic_group_action(g, xs)),
            assert not torch.allclose(x, torch.zeros_like(x), atol=1e-5), \
                f"Bad test: {layer_name} outputs 0 with group element {g}."
            assert torch.allclose(x, y, atol=1e-5), \
                f"{layer_name} doesn't commute with group action, group element: {g}."
    print(f"Passed test of {layer_name}")

def test_equi_flattened_isotypic_to_flattened_isotypic(
    layer, layer_name, irrep_size=[32, 196, 12], device="cuda"):
    with torch.inference_mode():
        xs = tuple(torch.randn(irrep_size, device=device) for _ in range(8))
        for g in group_elements:
            for x, y, irrep in zip(
                spatial_and_isotypic_group_action(g, layer(xs)),
                layer(spatial_and_isotypic_group_action(g, xs)),
                irreps
            ):
                assert not torch.allclose(x, torch.zeros_like(x), atol=1e-5), \
                    f"Bad test: {layer_name} outputs 0 with irrep {irrep} and group element {g}."
                assert torch.allclose(x, y, atol=1e-5), \
                    f"{layer_name} doesn't commute with group action, irrep: {irrep}, group element: {g}."
    print(f"Passed test of {layer_name}")

def test_equi_lift_d8():
    device = "cuda"
    lift8 = LiftD8(3, 768, bias=True, kernel_size=16, stride=16).to(device)
    test_equi_img_to_isotypic(lift8, "LiftD8", device=device)

def test_equi_patch_embed_d8():
    device = "cuda"
    lift8 = PatchEmbedD8(flatten=True).to(device)
    test_equi_img_to_flattened_isotypic(lift8, "PatchEmbedD8", device=device)

def test_equi_attention_d8():
    device = "cuda"
    irrep_size = [32, 196, 64]
    att8 = AttentionD8(dim=8*irrep_size[-1]).to(device)
    test_equi_isotypic_to_isotypic(att8, "AttentionD8", irrep_size=irrep_size, device=device)

def test_equi_flattened_isotypic_to_img(layer, layer_name, dim=768, device="cuda"):
    if dim % 8 != 0: raise ValueError()
    with torch.inference_mode():
        xs = tuple(torch.randn(32, 196, dim//8, device=device) for _ in range(8))
        for g in group_elements:
            x = image_space_group_action(g, layer(convert_8tuple_to_5tuple(xs)))
            y = layer(convert_8tuple_to_5tuple(spatial_and_isotypic_group_action(g, xs)))
            assert not torch.allclose(x, torch.zeros_like(x), atol=1e-5), \
                f"Bad test: {layer_name} outputs 0 with group element {g}."
            assert torch.allclose(x, y, atol=1e-5), \
                f"{layer_name} doesn't commute with group action, group element: {g}."
    print(f"Passed test of {layer_name}")

def test_equi_iso_to_patch_d8():
    device = "cuda"
    dim = 768
    topatch = IsotypicToPatchD8(dim=768, patch_side=16, reshape_to_image=True).to(device)
    test_equi_flattened_isotypic_to_img(topatch, "IsotypicToPatchD8", dim=dim, device=device)

def test_equi_one_img_to_img(net, net_name, device="cuda"):
    B = 33
    with torch.inference_mode():
        imgs = torch.randn(B, 3, 224, 224, device=device)
        noise = torch.rand(B, 196, device=device)
        for g in tqdm(group_elements):
            img_out1 = image_space_group_action(g, net(imgs, noise))
            img_out2 = net(
                image_space_group_action(g, imgs),
                image_space_group_action(g, noise.reshape(B, 14, 14)).reshape(B, 196)
            )
            assert not torch.allclose(img_out1, torch.zeros_like(img_out1), atol=1e-4), \
                f"Bad test: {net_name} outputs 0 with group element {g}."
            assert not torch.allclose(img_out1, image_space_group_action("r", img_out1), atol=1e-4), \
                f"Bad test: {net_name} outputs rot symmetric output with group element {g}."
            # TODO: is the atol below suspiciously high or not?
            assert torch.allclose(img_out1, img_out2, atol=1e-4), \
                f"{net_name} doesn't commute with group action, group element: {g}."
    print(f"Passed test of {net_name}")

def test_equi_d8_block():
    device = "cuda"
    irrep_size = [32, 196, 96]
    b8 = BlockD8(dim=8*irrep_size[-1], num_heads=12).to(device)
    test_equi_isotypic_to_isotypic(b8, "BlockD8", irrep_size=irrep_size, device=device)

def test_invariance_img_to_logits(net, net_name, device="cuda"):
    B = 33
    with torch.inference_mode():
        imgs = torch.randn(B, 3, 224, 224, device=device)
        for g in tqdm(group_elements):
            out1 = net(imgs)
            out2 = net(image_space_group_action(g, imgs))
            out3 = net(imgs.flip(-3)) # Flipping color dimension should not preserve invariance
            assert not torch.allclose(out1, torch.zeros_like(out1), atol=1e-4), \
                f"Bad test: {net_name} outputs 0 with group element {g}."
            assert not torch.allclose(out1, out3, atol=1e-4), "Bad test, these should be different"
            assert torch.allclose(out1, out2, atol=1e-4), \
                f"{net_name} doesn't commute with group action, group element: {g}."
    print(f"Passed INVARIANCE test of {net_name}")

def test_invariance_deit_inv_early():
    device = 'cuda'
    net = OcticVisionTransformer(depth=4, embed_dim=768, invariant=True).to(device)
    test_invariance_img_to_logits(
        net, 'OcticVisionTransformer', device=device
    )
    
def test_equi_flattened_isotypic_to_img_5tuple(layer, layer_name, dim=8*256, device="cuda", atol=1e-5):
    if dim % 8 != 0: raise ValueError()
    with torch.inference_mode():
        xs = tuple(torch.randn(32, 196, dim//8, device=device) for _ in range(8))
        for g in group_elements:
            x = image_space_group_action(g, layer(convert_8tuple_to_5tuple(xs)))
            y = layer(convert_8tuple_to_5tuple(spatial_and_isotypic_group_action(g, xs)))
            assert not torch.allclose(x, torch.zeros_like(x), atol=atol), \
                f"Bad test: {layer_name} outputs 0 with group element {g}."
            assert torch.allclose(x, y, atol=atol), \
                f"{layer_name} doesn't commute with group action, group element: {g}."
    print(f"Passed test of {layer_name}")


def test_inv_linear_invariant():
    device = "cuda"
    net = LinearInvariant(8*256).to(device)
    f = lambda x: net(x).reshape(
        x[0].shape[0], 14, 14, x[0].shape[2]
    ).permute(0, 3, 1, 2)
    test_equi_flattened_isotypic_to_img_5tuple(
        f, "LinearInvariant", device=device)

def test_inv_power_spectrum_invariant():
    device = "cuda"
    net = PowerSpectrumInvariant(8*256).to(device)
    f = lambda x: net(x).reshape(
        x[0].shape[0], 14, 14, 6*x[0].shape[2]
    ).permute(0, 3, 1, 2)
    test_equi_flattened_isotypic_to_img_5tuple(
        f, "PowerSpectrumInvariant", device=device)

def test_inv_polynomial_invariant():
    device = "cuda"
    net = PolynomialInvariant(8*256).to(device)
    f = lambda x: net(x).reshape(
        x[0].shape[0], 14, 14, -1#32*x[0].shape[2]
    ).permute(0, 3, 1, 2)
    test_equi_flattened_isotypic_to_img_5tuple(
        f, "PolynomialInvariant", device=device, atol=1e-4)

def test_inv_thirdorder_invariant():
    device = "cuda"
    net = ThirdOrderInvariant(8*256).to(device)
    f = lambda x: net(x).reshape(
        x[0].shape[0], 14, 14, 15*x[0].shape[2]
    ).permute(0, 3, 1, 2)
    test_equi_flattened_isotypic_to_img_5tuple(
        f, "ThirdOrderInvariant", device=device)

def test_inv_max_filtering_invariant():
    device = "cuda"
    num_filters = 1024
    net = MaxFilteringInvariant(8*256, num_filters).to(device)
    f = lambda x: net(x).reshape(
        x[0].shape[0], 14, 14, num_filters
    ).permute(0, 3, 1, 2)
    test_equi_flattened_isotypic_to_img_5tuple(
        f, "MaxFilteringInvariant", device=device)

def test_inv_canonization_invariant():
    device = "cuda"
    net = CanonizationInvariant(8*256).to(device)
    f = lambda x: net(x).reshape(
        x[0].shape[0], 14, 14, 8*x[0].shape[2]
    ).permute(0, 3, 1, 2)
    test_equi_flattened_isotypic_to_img_5tuple(
        f, "CanonizationInvariant", device=device)

if __name__ == "__main__":
    test_group_action()
    test_image_space_group_action()
    test_spatial_and_isotypic_group_action()
    test_fourier_transforms_inverses()
    test_fourier_transforms()
    test_equi_gelu_d8()
    test_equi_linear_d8()
    test_equi_layernorm_d8()
    test_equi_mlp_d8()
    test_equi_lift_d8()
    test_equi_patch_embed_d8()
    test_equi_attention_d8()
    test_equi_iso_to_patch_d8()
    test_equi_d8_block()
    test_invariance_deit_inv_early()

    # invariants
    test_inv_linear_invariant()
    test_inv_power_spectrum_invariant()
    test_inv_polynomial_invariant()
    test_inv_thirdorder_invariant()
    test_inv_max_filtering_invariant()
    test_inv_canonization_invariant()

    print("All tests passed!")
