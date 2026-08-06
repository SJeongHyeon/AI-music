import pytest

from confidence import assess_confidence


@pytest.mark.parametrize(
    ("top3", "expected_level", "expected_margin"),
    [
        ([('rock', 0.70), ('metal', 0.20), ('pop', 0.10)], '높음', 0.50),
        ([('rock', 0.50), ('metal', 0.35), ('pop', 0.15)], '높음', 0.15),
        ([('pop', 0.42), ('disco', 0.30), ('rock', 0.10)], '보통', 0.12),
        ([('pop', 0.30), ('disco', 0.22), ('rock', 0.12)], '보통', 0.08),
        ([('reggae', 0.14), ('pop', 0.13), ('jazz', 0.13)], '낮음', 0.01),
    ],
)
def test_assess_confidence_levels(top3, expected_level, expected_margin):
    result = assess_confidence(top3)

    assert result['level'] == expected_level
    assert result['margin'] == expected_margin
    assert result['top1_prob'] == top3[0][1]


def test_assess_confidence_requires_two_predictions():
    with pytest.raises(ValueError, match='최소 2개'):
        assess_confidence([('rock', 0.70)])


def test_assess_confidence_does_not_mutate_predictions():
    top3 = [('reggae', 0.14), ('pop', 0.13), ('jazz', 0.13)]
    original = list(top3)

    assess_confidence(top3)

    assert top3 == original
