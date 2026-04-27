from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200
from .mobilefacenet import get_mbf, get_mbf_large, get_mbf_large_v2, get_mbf_v3, get_mbf_v3_se, get_mbf_v3_se_qat
from .fused_mobilenet_v2_se import get_fused_mobilent_v2_se
from .fused_efficient_mbn_v2_se_256d_mutant_a1 import get_efficient_mbn_v2_se_mutant_a1
from .fused_optimized_irse_skip_se_v1_mut1_5 import get_irse_skip_se_v1_mut1_5


def get_model(name, **kwargs):
    # resnet
    if name == "r18":
        return iresnet18(False, **kwargs)
    elif name == "r34":
        return iresnet34(False, **kwargs)
    elif name == "r50":
        return iresnet50(False, **kwargs)
    elif name == "r100":
        return iresnet100(False, **kwargs)
    elif name == "r200":
        return iresnet200(False, **kwargs)
    elif name == "r2060":
        from .iresnet2060 import iresnet2060
        return iresnet2060(False, **kwargs)

    elif name == "mbf":
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return get_mbf(fp16=fp16, num_features=num_features)

    elif name == "mbf_large":
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return get_mbf_large(fp16=fp16, num_features=num_features)
    
    elif name == "mbf_large_v2":
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return get_mbf_large_v2(fp16=fp16, num_features=num_features)

    elif name == "mbf_v3":
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return get_mbf_v3(fp16=fp16, num_features=num_features)

    elif name == "mbf_v3_se":
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        se_reduction = kwargs.get("se_reduction", 16)
        return get_mbf_v3_se(fp16=fp16, num_features=num_features, se_reduction=se_reduction)

    elif name == "mbf_v3_se_qat":
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        se_reduction = kwargs.get("se_reduction", 16)
        return get_mbf_v3_se_qat(fp16=fp16, num_features=num_features, se_reduction=se_reduction)

    elif name == "fused_mbn_v2_se":
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        channels = kwargs.get("channels", (64, 128, 256))
        use_se = kwargs.get("use_se", True)
        return get_fused_mobilent_v2_se(fp16=fp16, num_features=num_features,
                                         channels=channels, use_se=use_se)

    elif name == "femv2_se_mutant_a1":
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return get_efficient_mbn_v2_se_mutant_a1(fp16=fp16, num_features=num_features)

    elif name == "irse_skip_se_v1_mut1_5":
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return get_irse_skip_se_v1_mut1_5(fp16=fp16, num_features=num_features)

    elif name == "vit_t":
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=256, depth=12,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1)

    elif name == "vit_t_dp005_mask0": # For WebFace42M
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=256, depth=12,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.0)

    elif name == "vit_s":
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=12,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1)
    
    elif name == "vit_s_dp005_mask_0":  # For WebFace42M
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=12,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.0)
    
    elif name == "vit_b":
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=24,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1, using_checkpoint=True)

    elif name == "vit_b_dp005_mask_005":  # For WebFace42M
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=24,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.05, using_checkpoint=True)

    elif name == "vit_l_dp005_mask_005":  # For WebFace42M
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=768, depth=24,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.05, using_checkpoint=True)
        
    elif name == "vit_h":  # For WebFace42M
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=1024, depth=48,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0, using_checkpoint=True)

    else:
        raise ValueError()
