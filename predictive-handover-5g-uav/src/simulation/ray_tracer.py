import numpy as np
import tensorflow as tf
from sionna.rt import PathSolver, Receiver

class BatchRayTracer:
    """Executes parallel GPU ray-tracing and converts CIR tensors to wideband RSRP."""
    def __init__(self, scene_builder, config: dict):
        self.scene = scene_builder.scene
        self.config = config
        self.solver = PathSolver()
        self.noise_floor = float(self.config['simulation']['thermal_noise_floor_dbm'])

    def compute_trajectory_rsrp(self, trajectory_coords: np.ndarray) -> np.ndarray:
        """
        Computes 28 GHz RSRP across all waypoints simultaneously on the GPU.
        trajectory_coords: shape (N_steps, 3) representing [x, y, z] in meters.
        Returns: RSRP matrix of shape (N_steps, num_gnbs) in dBm.
        """
        # Clear previous receivers
        for name in list(self.scene.receivers.keys()):
            self.scene.remove(name)

        num_steps = len(trajectory_coords)
        for s in range(num_steps):
            rx = Receiver(
                name=f"uav_rx_{s}",
                position=[float(c) for c in trajectory_coords[s]]
            )
            self.scene.add(rx)

        # Execute parallel ray tracing
        paths = self.solver(
            scene=self.scene,
            max_depth=self.config['simulation']['max_depth'],
            los=True,
            specular_reflection=True
        )

        try:
            a, _ = paths.cir(out_type="numpy")
        except TypeError:
            a, _ = paths.cir()

        a_arr = np.asarray(a) # Shape: [N_rx, rx_ant, N_tx, tx_ant, paths, time_steps]
        power = np.abs(a_arr) ** 2
        
        # Sum energy across all dimensions except RX waypoints (axis 0) and TX gNBs (axis 2)
        axes_to_sum = tuple(ax for ax in range(a_arr.ndim) if ax not in (0, 2))
        p_total_watts = np.sum(power, axis=axes_to_sum)
        
        # Convert to dBm with -140 dBm noise floor
        p_clamped = np.maximum(p_total_watts, 10 ** (self.noise_floor / 10.0))
        rsrp_dbm = 10.0 * np.log10(p_clamped) + 30.0
        return rsrp_dbm