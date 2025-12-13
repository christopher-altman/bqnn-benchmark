from bqnn.sweeps import SweepConfig, SweepSpace, run_sweep


def test_sweep_smoke(tmp_path):
    cfg = SweepConfig(
        name="smoke_sweep",
        output_dir=tmp_path,
        n_features=8,
        n_hidden=4,
        n_classes=2,
        n_train=128,
        n_test=64,
        batch_size=32,
        epochs=1,
        device="cpu",
    )
    space = SweepSpace(a=[0.0, 0.5], lr=[1e-3], noise_pairs=[0], noise_angle=[0.0])

    rows = run_sweep(cfg, space, search="grid", seed=0, make_plots=False)
    assert len(rows) == 2
    assert all("accuracy" in r for r in rows)
