import numpy as np

# Flow telemetry is extremely heavy-tailed: Flow Duration and byte counters span
# ~8 orders of magnitude, so a plain StandardScaler leaves 99% of windows squashed
# near zero with a handful of 100-sigma outliers. A signed log1p compression before
# standardisation is what keeps the LSTM out of saturation.
class LogStandardScaler:
    """signed-log1p compression followed by per-feature standardisation."""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    @staticmethod
    def _compress(X):
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return np.sign(X) * np.log1p(np.abs(X))

    def fit(self, X):
        Z = self._compress(X)
        self.mean_ = Z.mean(axis=0)
        std = Z.std(axis=0)
        # Constant features get scale 1 so they map to exactly 0 instead of NaN.
        self.scale_ = np.where(std < 1e-8, 1.0, std)
        return self

    def transform(self, X):
        Z = self._compress(X)
        return ((Z - self.mean_) / self.scale_).astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        Y = np.asarray(Z, dtype=np.float64) * self.scale_ + self.mean_
        return np.sign(Y) * np.expm1(np.abs(Y))

    # --- plain-array serialisation (avoids pickling library objects) --- #
    def state_dict(self):
        return {"mean": np.asarray(self.mean_), "scale": np.asarray(self.scale_)}

    @classmethod
    def from_state_dict(cls, d):
        obj = cls()
        obj.mean_ = np.asarray(d["mean"], dtype=np.float64)
        obj.scale_ = np.asarray(d["scale"], dtype=np.float64)
        return obj
