import numpy as np

class Standard3GPPA3Handover:
    """Simulates 3GPP Rel-15/16 Event-A3 Handover State Machine."""
    def __init__(self, hom_db: float = 3.0, ttt_steps: int = 4, exec_delay_steps: int = 1, outage_thresh: float = -95.0):
        self.hom = hom_db
        self.ttt_steps = ttt_steps
        self.exec_delay = exec_delay_steps
        self.outage_thresh = outage_thresh

    def simulate(self, rsrp_matrix: np.ndarray, init_serving: int = 0) -> dict:
        num_steps = len(rsrp_matrix)
        serving = init_serving
        serving_history = np.zeros(num_steps, dtype=int)
        active_rsrp = np.zeros(num_steps)
        
        ttt_counter = 0
        candidate_cell = None
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
            else:
                nbr_rsrps = rsrp_matrix[t].copy()
                nbr_rsrps[serving] = -np.inf
                best_nbr = int(np.argmax(nbr_rsrps))

                # 3GPP Event A3 entering condition
                if (nbr_rsrps[best_nbr] - s_rsrp) > self.hom:
                    if candidate_cell == best_nbr:
                        ttt_counter += 1
                    else:
                        candidate_cell = best_nbr
                        ttt_counter = 1

                    if ttt_counter >= self.ttt_steps:
                        pending_target = candidate_cell
                        exec_timer = self.exec_delay
                        candidate_cell = None
                        ttt_counter = 0
                else:
                    candidate_cell = None
                    ttt_counter = 0

        return {
            "serving_history": serving_history,
            "active_rsrp": active_rsrp,
            "ho_events": ho_events,
            "rlf_steps": rlf_steps,
            "ping_pong_count": ping_pong_count
        }