from .base import BaseMatcher, MatchResult


def build_matcher(args) -> BaseMatcher:
    matcher = args.matcher.lower()
    if matcher == "sift":
        from .sift import SIFTMatcher
        return SIFTMatcher(
            ratio_thresh=args.sift_ratio_thresh,
            fm_ransac_thresh=args.fm_ransac_thresh,
            fm_confidence=args.fm_confidence,
            device=args.device,
        )
    if matcher == "loftr":
        from .loftr import LoFTRMatcher
        return LoFTRMatcher(
            weight_path=args.loftr_weight_path,
            fm_ransac_thresh=args.fm_ransac_thresh,
            fm_confidence=args.fm_confidence,
            device=args.device,
        )
    if matcher == "superglue":
        from .superglue import SuperPointSuperGlueMatcher
        return SuperPointSuperGlueMatcher(
            superpoint_weight_path=args.superpoint_weight_path,
            superglue_weight_path=args.superglue_weight_path,
            device=args.device,
        )
    if matcher == "aspanformer":
        if not args.aspanformer_config_path:
            raise ValueError("ASpanFormer config path is required")
        if not args.aspanformer_weight_path:
            raise ValueError("ASpanFormer weight path is required")
        from .aspanformer import ASpanFormerMatcher
        return ASpanFormerMatcher(
            config_path=args.aspanformer_config_path,
            weight_path=args.aspanformer_weight_path,
            device=args.device,
        )
    if matcher == "roma":
        from .roma import RoMaMatcher
        return RoMaMatcher(
            variant=args.roma_variant,
            weight_path=args.roma_weight_path,
            device=args.device,
        )
    raise ValueError(f"Unsupported matcher: {args.matcher}")
