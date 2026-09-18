import numpy as np

class UAVTrajectoryGenerator:
    """Generates continuous 3D UAV kinematics for training diversity and knife-edge stress tests."""
    @staticmethod
    def generate_random_flight(steps: int, corridor_type: str = "corner", seed: int = 42) -> np.ndarray:
        np.random.seed(seed)
        if corridor_type == "corner":
            start_x, end_x = np.random.uniform(-125.0, -115.0), np.random.uniform(-65.0, -55.0)
            start_y, end_y = np.random.uniform(55.0, 60.0), np.random.uniform(20.0, 30.0)
            start_z, end_z = np.random.uniform(16.0, 20.0), np.random.uniform(18.0, 22.0)
        else: # Boulevard flight
            start_x, end_x = np.random.uniform(-80.0, -50.0), np.random.uniform(40.0, 70.0)
            start_y, end_y = np.random.uniform(-30.0, 30.0), np.random.uniform(-30.0, 30.0)
            start_z, end_z = np.random.uniform(18.0, 25.0), np.random.uniform(20.0, 30.0)

        x = np.linspace(start_x, end_x, steps)
        y = np.linspace(start_y, end_y, steps)
        z = np.linspace(start_z, end_z, steps)
        return np.column_stack([x, y, z])

    @staticmethod
    def generate_knife_edge_flight(num_pts: int = 50) -> np.ndarray:
        """Physical trajectory traversing from LoS into the high-rise shadow boundary."""
        x = np.linspace(-120.0, -100.0, num_pts)
        y = np.linspace(58.0, 48.0, num_pts)
        z = np.full(num_pts, 16.0)
        return np.column_stack([x, y, z])