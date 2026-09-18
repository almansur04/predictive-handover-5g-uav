import tensorflow as tf
import sionna as sn
from sionna.rt import load_scene, PlanarArray, Transmitter

class SionnaSceneBuilder:
    """Builds and manages the 3D 28 GHz ray-tracing environment in NVIDIA Sionna."""
    def __init__(self, config: dict):
        self.config = config
        self._configure_gpu()
        
        # Load 3D urban microcell
        self.scene = load_scene(sn.rt.scene.etoile)
        self.scene.frequency = float(self.config['simulation']['carrier_frequency'])
        self.scene.synthetic_array = True
        
        self._configure_antenna_arrays()
        self._deploy_base_stations()

    def _configure_gpu(self):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"[!] GPU Memory config notice: {e}")

    def _configure_antenna_arrays(self):
        # 4x4 Planar Array (16 elements, vertical polarization) for gNBs
        self.scene.tx_array = PlanarArray(
            num_rows=4,
            num_cols=4,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization="V"
        )
        # Single isotropic antenna for UAV
        self.scene.rx_array = PlanarArray(
            num_rows=1,
            num_cols=1,
            pattern="iso",
            polarization="V"
        )

    def _deploy_base_stations(self):
        # Reset existing transmitters and receivers
        for name in list(self.scene.transmitters.keys()):
            self.scene.remove(name)
        for name in list(self.scene.receivers.keys()):
            self.scene.remove(name)

        pwr = float(self.config['simulation']['tx_power_dbm'])
        for idx, pos in enumerate(self.config['gnb_positions']):
            tx = Transmitter(name=f"gnb_{idx}", position=pos, power_dbm=pwr)
            self.scene.add(tx)