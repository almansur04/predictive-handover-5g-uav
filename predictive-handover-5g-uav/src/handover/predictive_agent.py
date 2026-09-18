import numpy as np

class PredictiveHandoverAgent:
    """Proactive mobility controller using Deep Learning sequence forecasting."""
    def __init__(self, model, scaler, conf_threshold: float = 0.70, exec_delay_steps: int = 1, outage_thresh: float = -95.0):
        self.model = model
        self.scaler = scaler
        self.conf_threshold = conf_threshold
        self.exec_delay = exec_delay_steps
        self.outage_thresh = outage_thresh

    def simulate(self, rsrp_matrix: np.ndarray, coords_matrix: np.ndarray, init_serving: int = 0) -> dict:
        num_steps = len(rsrp_matrix)
        serving = init_serving
        features = np.hstack([coords_matrix, rsrp_matrix])
        
        serving_history = np.zeros(num_steps, dtype=int)
        active_rsrp = np.zeros(num_steps)
        
        pending_target = None
        exec_timer = 0
        ho_events = []
        rlf_steps = 0
        ping_pong_count = 0
        prev_serving = None
        last_ho_step = -999

        for t in range(num_steps):
            s_rsrp = rsrp_matrix[t, serving]
            active_rsrp[t] = s_rsrp
            serving_history[t] = serving

            if s_rsrp < self.outage_thresh:
                rlf_steps += 1

            if pending_target is not None:
                exec_timer -= 1
                if exec_timer <= 0:
                    if prev_serving == pending_target and (t - last_ho_step) <= 20:
                        ping_pong_count += 1
                    prev_serving = serving
                    serving = pending_target
                    pending_target = None
                    ho_events.append((t, prev_serving, serving))
                    last_ho_step = t
            elif t >= 10:
                win = features[t-10:t, :]
                win_scaled = self.scaler.transform(win).reshape(1, 10, -1)
                probs = self.model(win_scaled, training=False).numpy()[0]
                pred_target = int(np.argmax(probs))

                if pred_target != serving and probs[pred_target] >= self.conf_threshold:
                    pending_target = pred_target
                    exec_timer = self.exec_delay

        return {
            "serving_history": serving_history,
            "active_rsrp": active_rsrp,
            "ho_events": ho_events,
            "rlf_steps": rlf_steps,
            "ping_pong_count": ping_pong_count
        }