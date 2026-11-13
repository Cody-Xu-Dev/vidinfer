"""分块误差累计；NumPy 延迟导入，验证不在计时区间内。"""

import math
import statistics


def summarize(values):
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Timing samples must be finite and nonnegative")
    ordered = sorted(values)

    def quantile(p):
        position = (len(ordered) - 1) * p
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return dict(
        count=len(values),
        samples_ms=values,
        median_ms=statistics.median(values),
        p10_ms=quantile(0.1),
        p90_ms=quantile(0.9),
        min_ms=min(values),
        max_ms=max(values),
    )


def compare_chunks(chunks, *, mode, atol, rtol):
    """chunks 产生同形状的 (reference, candidate) 数组，不一次拷回巨大输出。

    相对 L2 在零参考且非零误差时无定义，返回 null；余弦在零范数时也返回 null。
    exact 检查浮点位模式（包括正负零），allclose 使用逐元素 atol + rtol*abs(ref)。
    """
    import numpy as np

    total = different = failed = 0
    finite = exact = True
    maximum = abs_sum = error_sq = ref_sq = got_sq = dot = 0.0
    for reference, candidate in chunks:
        a = np.asarray(reference, dtype=np.float64).reshape(-1)
        b = np.asarray(candidate, dtype=np.float64).reshape(-1)
        if a.shape != b.shape:
            raise ValueError("Mismatched output chunks")
        total += a.size
        finite = finite and bool(np.isfinite(a).all() and np.isfinite(b).all())
        exact = exact and bool(np.array_equal(a.view(np.uint64), b.view(np.uint64)))
        if not finite:
            continue
        delta = b - a
        absolute = np.abs(delta)
        maximum = max(maximum, float(absolute.max(initial=0)))
        abs_sum += float(absolute.sum())
        error_sq += float(np.dot(delta, delta))
        ref_sq += float(np.dot(a, a))
        got_sq += float(np.dot(b, b))
        dot += float(np.dot(a, b))
        different += int(np.count_nonzero(a != b))
        failed += int(np.count_nonzero(absolute > atol + rtol * np.abs(a)))
    if not total:
        raise ValueError("Empty output is not a valid correctness check")
    numeric = finite and all(
        math.isfinite(v) for v in (maximum, abs_sum, error_sq, ref_sq, got_sq, dot)
    )
    cosine = dot / math.sqrt(ref_sq * got_sq) if numeric and ref_sq and got_sq else None
    relative = (
        math.sqrt(error_sq / ref_sq)
        if numeric and ref_sq
        else (0.0 if numeric and error_sq == 0 else None)
    )
    return dict(
        elements=int(total),
        finite=finite,
        bitwise_exact=finite and exact,
        different_elements=different if finite else None,
        failed_elements=failed if finite else None,
        max_absolute_error=maximum if numeric else None,
        mean_absolute_error=abs_sum / total if numeric else None,
        relative_l2=relative,
        cosine_similarity=max(-1.0, min(1.0, cosine)) if cosine is not None else None,
        passes=bool(numeric and (exact if mode == "exact" else failed == 0)),
    )


def compare_outputs(reference, candidate, spec, chunk_elements=2**20):
    """算子统一返回具名 Tensor 字典；QK 的两个输出分别验收。"""
    if reference.keys() != candidate.keys():
        raise ValueError("Output keys differ")
    results = {}
    for name, a in reference.items():
        b = candidate[name]
        if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
            raise ValueError(f"Output shape/dtype/device mismatch: {name}")
        # 统一算子边界要求 contiguous；这里不能偷偷修补候选的输出布局。
        if not a.is_contiguous() or not b.is_contiguous():
            raise ValueError(
                "Candidate must return contiguous outputs inside its timed call"
            )
        av, bv = a.view(-1), b.view(-1)

        def chunks():
            for start in range(0, a.numel(), chunk_elements):
                yield (
                    av[start : start + chunk_elements].double().cpu().numpy(),
                    bv[start : start + chunk_elements].double().cpu().numpy(),
                )

        results[name] = compare_chunks(
            chunks(), mode=spec.mode, atol=spec.atol, rtol=spec.rtol
        )
    return dict(
        passes=all(item["passes"] for item in results.values()), outputs=results
    )
