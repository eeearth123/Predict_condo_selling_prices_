
import numpy as np

class TwoSegmentRegressor:
    def __init__(
        self,
        mass_encoder, mass_model,
        lux_encoder, lux_model,
        cat_cols, selected_features,
        threshold=100
    ):
        self.mass_encoder = mass_encoder
        self.mass_model = mass_model
        self.lux_encoder = lux_encoder
        self.lux_model = lux_model
        self.cat_cols = cat_cols
        self.selected_features = selected_features
        self.threshold = threshold

    def _transform(self, encoder, X):
        X = X.copy()
        if self.cat_cols:
            X[self.cat_cols] = encoder.transform(X[self.cat_cols])
        return X

    def predict(self, X):
        X = X.copy()
        X = X[self.selected_features]
        use_lux = X["Area_sqm"] > 250

        preds = np.zeros(len(X), dtype=float)

        if (~use_lux).any():
            Xm = X[~use_lux]
            Xm = self._transform(self.mass_encoder, Xm)
            preds[~use_lux] = self.mass_model.predict(Xm)

        if use_lux.any():
            Xl = X[use_lux]
            Xl = self._transform(self.lux_encoder, Xl)
            preds[use_lux] = self.lux_model.predict(Xl)

        return preds
