import numpy as np
from sklearn.preprocessing import StandardScaler

class TelemetryDataPipeline:
    """Processes trajectory features into sliding window tensors and normalized training samples."""
    def __init__(self, lookback_w: int = 10, pred_horizon: int = 5):
        self.lookback_w = lookback_w
        self.pred_horizon = pred_horizon
        self.scaler = StandardScaler()

    def create_sliding_windows(self, coords_list: list, rsrp_list: list):
        X_samples, y_samples = [], []
        
        for coords, rsrp in zip(coords_list, rsrp_list):
            features = np.hstack([coords, rsrp])
            num_steps = len(features)
            
            for t in range(self.lookback_w, num_steps - self.pred_horizon):
                window = features[t - self.lookback_w : t, :]
                target_cell = int(np.argmax(rsrp[t + self.pred_horizon, :]))
                
                X_samples.append(window)
                y_samples.append(target_cell)
                
        X_arr = np.array(X_samples, dtype=np.float32)
        y_arr = np.array(y_samples, dtype=np.int32)
        
        # Fit scaler on flattened 2D feature matrix, then restore 3D sequence shape
        N, W, F = X_arr.shape
        X_scaled = self.scaler.fit_transform(X_arr.reshape(-1, F)).reshape(N, W, F)
        return X_scaled, y_arr