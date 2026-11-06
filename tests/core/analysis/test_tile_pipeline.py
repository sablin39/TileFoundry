import pytest

from tilefoundry.analysis import TileStage, tile_pipeline


def test_tiles_cross_three_resources_and_keep_the_phase_fence():
    stages = (
        TileStage("receive", "network", 1),
        TileStage("relay", "link", 1),
        TileStage("use", "mma", 1),
    )
    phases = (stages, (TileStage("consume", "mma", 2),))
    bulk = tile_pipeline(phases, tiles=4, slots=2, policy="bulk")
    tiled = tile_pipeline(phases, tiles=4, slots=2, policy="tile_ready")
    assert bulk[-1].end == 20 and tiled[-1].end == 14
    assert [s.start for s in tiled if s.stage.name == "consume"] == [6, 8, 10, 12]
    assert {(s.stage.name, s.tile) for s in tiled if s.start <= 2 < s.end} == {
        ("receive", 2),
        ("relay", 1),
        ("use", 0),
    }


def test_reuse_waits_for_consumption_and_shared_resources_serialize():
    stages = (TileStage("receive", "link", 2), TileStage("consume", "mma", 5))
    one = tile_pipeline((stages,), tiles=3, slots=1, policy="tile_ready")
    two = tile_pipeline((stages,), tiles=3, slots=2, policy="tile_ready")
    assert [s.start for s in one if s.stage == stages[0]] == [0, 7, 14]
    assert [s.start for s in two if s.stage == stages[0]] == [0, 2, 7]
    same = (TileStage("first", "device", 2), TileStage("second", "device", 5))
    serial = tile_pipeline((same,), tiles=3, slots=2, policy="tile_ready")
    assert serial[-1].end == 21
    assert tile_pipeline((same,), tiles=0, slots=1, policy="tile_ready") == ()
    with pytest.raises(ValueError, match="slots"):
        tile_pipeline((same,), tiles=1, slots=0, policy="tile_ready")
    with pytest.raises(ValueError, match="duration"):
        tile_pipeline(
            ((TileStage("transfer", "link", float("nan")),),), tiles=1, slots=1, policy="bulk"
        )
