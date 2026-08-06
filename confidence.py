from collections.abc import Sequence


def assess_confidence(
    top3: Sequence[tuple[str, float]],
) -> dict[str, str | float]:
    """Top-3 예측을 UI 설명용 상대적 신뢰도로 변환한다."""
    if len(top3) < 2:
        raise ValueError('신뢰도 판단에는 최소 2개 예측이 필요합니다.')

    top1_prob = float(top3[0][1])
    top2_prob = float(top3[1][1])
    margin = round(top1_prob - top2_prob, 4)

    if top1_prob >= 0.50 and margin >= 0.15:
        level = '높음'
        message = '현재 지원 장르 안에서 비교적 뚜렷한 결과입니다.'
        color = 'green'
    elif top1_prob >= 0.30 and margin >= 0.08:
        level = '보통'
        message = '가능성은 있지만 다른 후보와 함께 확인해야 합니다.'
        color = 'orange'
    else:
        level = '낮음'
        message = '결과를 확정하기 어렵습니다.'
        color = 'red'

    return {
        'level': level,
        'top1_prob': top1_prob,
        'margin': margin,
        'message': message,
        'color': color,
    }
