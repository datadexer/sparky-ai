import pandas as pd
from sparky.validation.parameter_plateau import (
    parameter_plateau_test,
    parameter_sensitivity_1d,
)


def _make_df(sharpes, param_values=None):
    if param_values is None:
        param_values = list(range(len(sharpes)))
    return pd.DataFrame({"param": param_values, "sharpe": sharpes})


class TestParameterPlateauTest:
    def test_flat_landscape_passes(self):
        df = _make_df([1.0, 1.01, 0.99, 1.02, 0.98])
        result = parameter_plateau_test(df)
        assert result.passed
        assert result.coverage > 0.9

    def test_single_spike_fails(self):
        sharpes = [0.1] * 9 + [2.0]
        df = _make_df(sharpes)
        result = parameter_plateau_test(df)
        assert not result.passed
        assert result.n_in_plateau == 1

    def test_smooth_hill_passes(self):
        sharpes = [0.8, 0.9, 1.0, 1.1, 1.2, 1.15, 1.0, 0.9, 0.85, 0.8]
        df = _make_df(sharpes)
        result = parameter_plateau_test(df, threshold_frac=0.30)
        assert result.passed

    def test_single_config_passes(self):
        df = _make_df([1.5])
        result = parameter_plateau_test(df)
        assert result.passed
        assert result.coverage == 1.0


class TestParameterSensitivity1d:
    def test_monotonic_increasing(self):
        df = _make_df([0.5, 1.0, 1.5], param_values=[10, 20, 30])
        result = parameter_sensitivity_1d(df, "param")
        assert result["is_monotonic"]

    def test_non_monotonic(self):
        df = _make_df([0.5, 1.5, 0.5], param_values=[10, 20, 30])
        result = parameter_sensitivity_1d(df, "param")
        assert not result["is_monotonic"]

    def test_single_value(self):
        df = _make_df([1.0], param_values=[10])
        result = parameter_sensitivity_1d(df, "param")
        assert result["is_monotonic"]
        assert result["max_gradient"] == 0.0
