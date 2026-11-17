"""Final refits may consume the old validation fold only after excluding test games."""

import pytest
from refit_external import validated_refit_games


@pytest.mark.parametrize(
    "games,allowed",
    [
        (["train", "val"], True),
        (["train", "test"], False),
        (["train", "train", "val"], False),
        (["train"], False),
    ],
)
def test_refit_uses_exactly_original_train_and_validation_without_test(games, allowed):
    selection = {
        "train_games": ["train"],
        "validation_games": ["val"],
        "test_games": ["test"],
        "stride": 5,
        "decoder_scales": [0.3],
    }
    if allowed:
        assert validated_refit_games(selection, games) == games
    else:
        with pytest.raises(ValueError, match="exclude every test"):
            validated_refit_games(selection, games)
