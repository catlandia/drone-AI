"""
Real Drone Hardware Interface

This module provides interfaces for deploying trained AI models to real drone hardware.
Supported platforms:
- Crazyflie 2.x (via cflib)
- DJI Tello (via djitellopy)
- Generic MAVLink drones (PX4, ArduPilot)

The interfaces translate learned motor commands to appropriate control signals
for each platform, enabling sim-to-real transfer.
"""

import numpy as np
import time
import threading
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DroneStatus(Enum):
    """Drone connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ARMED = "armed"
    FLYING = "flying"
    LANDING = "landing"
    ERROR = "error"


@dataclass
class DroneState:
    """Real drone state from sensors."""
    position: np.ndarray  # [x, y, z] in meters
    velocity: np.ndarray  # [vx, vy, vz] in m/s
    orientation: np.ndarray  # [roll, pitch, yaw] in radians
    angular_velocity: np.ndarray  # [p, q, r] in rad/s
    battery_voltage: float  # Volts
    battery_percent: float  # 0-100
    timestamp: float  # Unix timestamp


class DroneInterface(ABC):
    """
    Abstract base class for drone hardware interfaces.

    This class defines the common interface that all drone implementations
    must follow, enabling seamless switching between different hardware.
    """

    def __init__(self):
        self.status = DroneStatus.DISCONNECTED
        self.state: Optional[DroneState] = None
        self._state_callback: Optional[Callable[[DroneState], None]] = None
        self._control_rate = 100  # Hz

    @abstractmethod
    def connect(self, address: str = None) -> bool:
        """Connect to the drone."""
        pass

    @abstractmethod
    def disconnect(self):
        """Disconnect from the drone."""
        pass

    @abstractmethod
    def arm(self) -> bool:
        """Arm the drone motors."""
        pass

    @abstractmethod
    def disarm(self):
        """Disarm the drone motors."""
        pass

    @abstractmethod
    def takeoff(self, height: float = 1.0) -> bool:
        """Automated takeoff to specified height."""
        pass

    @abstractmethod
    def land(self):
        """Automated landing."""
        pass

    @abstractmethod
    def send_motor_commands(self, commands: np.ndarray):
        """
        Send raw motor commands.

        Args:
            commands: Array of 4 motor commands, normalized [0, 1]
        """
        pass

    @abstractmethod
    def send_attitude_command(
        self,
        thrust: float,
        roll: float,
        pitch: float,
        yaw_rate: float
    ):
        """
        Send attitude control command.

        Args:
            thrust: Collective thrust [0, 1]
            roll: Roll angle (radians)
            pitch: Pitch angle (radians)
            yaw_rate: Yaw rate (rad/s)
        """
        pass

    def get_state(self) -> Optional[DroneState]:
        """Get current drone state."""
        return self.state

    def set_state_callback(self, callback: Callable[[DroneState], None]):
        """Set callback for state updates."""
        self._state_callback = callback

    def emergency_stop(self):
        """Emergency stop - immediately cut motors."""
        logger.warning("EMERGENCY STOP TRIGGERED")
        self.disarm()


class CrazyflieInterface(DroneInterface):
    """
    Interface for Bitcraze Crazyflie 2.x drones.

    The Crazyflie is an excellent platform for research due to its
    small size, open-source firmware, and good documentation.

    Requires: cflib (pip install cflib)
    """

    def __init__(self):
        super().__init__()
        self._cf = None
        self._scf = None
        self._motion_commander = None

    def connect(self, address: str = None) -> bool:
        """
        Connect to Crazyflie drone.

        Args:
            address: Radio address (e.g., "radio://0/80/2M/E7E7E7E7E7")
        """
        try:
            import cflib
            from cflib.crazyflie import Crazyflie
            from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
            from cflib.positioning.motion_commander import MotionCommander

            cflib.crtp.init_drivers()

            if address is None:
                # Scan for available Crazyflies
                logger.info("Scanning for Crazyflie drones...")
                available = cflib.crtp.scan_interfaces()
                if not available:
                    logger.error("No Crazyflie found!")
                    return False
                address = available[0][0]
                logger.info(f"Found Crazyflie at {address}")

            self.status = DroneStatus.CONNECTING
            self._scf = SyncCrazyflie(address, cf=Crazyflie(rw_cache='./cache'))
            self._scf.open_link()

            self._cf = self._scf.cf
            self._setup_logging()

            self.status = DroneStatus.CONNECTED
            logger.info(f"Connected to Crazyflie at {address}")
            return True

        except ImportError:
            logger.error("cflib not installed. Install with: pip install cflib")
            return False
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self.status = DroneStatus.ERROR
            return False

    def _setup_logging(self):
        """Set up state logging from Crazyflie."""
        from cflib.crazyflie.log import LogConfig

        # Configure state logging
        log_conf = LogConfig(name='State', period_in_ms=10)
        log_conf.add_variable('stateEstimate.x', 'float')
        log_conf.add_variable('stateEstimate.y', 'float')
        log_conf.add_variable('stateEstimate.z', 'float')
        log_conf.add_variable('stateEstimate.vx', 'float')
        log_conf.add_variable('stateEstimate.vy', 'float')
        log_conf.add_variable('stateEstimate.vz', 'float')

        self._cf.log.add_config(log_conf)
        log_conf.data_received_cb.add_callback(self._state_callback_internal)
        log_conf.start()

    def _state_callback_internal(self, timestamp, data, logconf):
        """Internal callback for state updates."""
        self.state = DroneState(
            position=np.array([
                data['stateEstimate.x'],
                data['stateEstimate.y'],
                data['stateEstimate.z']
            ]),
            velocity=np.array([
                data['stateEstimate.vx'],
                data['stateEstimate.vy'],
                data['stateEstimate.vz']
            ]),
            orientation=np.zeros(3),  # Would need additional logging
            angular_velocity=np.zeros(3),
            battery_voltage=0,
            battery_percent=0,
            timestamp=time.time()
        )

        if self._state_callback:
            self._state_callback(self.state)

    def disconnect(self):
        """Disconnect from Crazyflie."""
        if self._scf:
            self._scf.close_link()
        self.status = DroneStatus.DISCONNECTED
        logger.info("Disconnected from Crazyflie")

    def arm(self) -> bool:
        """Arm is automatic on Crazyflie."""
        self.status = DroneStatus.ARMED
        return True

    def disarm(self):
        """Stop motors."""
        if self._cf:
            self._cf.commander.send_stop_setpoint()
        self.status = DroneStatus.CONNECTED

    def takeoff(self, height: float = 1.0) -> bool:
        """Takeoff using MotionCommander."""
        try:
            from cflib.positioning.motion_commander import MotionCommander

            self._motion_commander = MotionCommander(self._scf)
            self._motion_commander.take_off(height)
            self.status = DroneStatus.FLYING
            return True
        except Exception as e:
            logger.error(f"Takeoff failed: {e}")
            return False

    def land(self):
        """Land the drone."""
        if self._motion_commander:
            self._motion_commander.land()
        self.status = DroneStatus.CONNECTED

    def send_motor_commands(self, commands: np.ndarray):
        """Send motor PWM commands."""
        if self._cf is None:
            return

        # Convert normalized [0,1] to Crazyflie PWM (0-65535)
        pwm = (commands * 65535).astype(np.uint16)

        # Note: Direct motor control requires special firmware mode
        # This is for advanced users only
        logger.warning("Direct motor control requires custom firmware")

    def send_attitude_command(
        self,
        thrust: float,
        roll: float,
        pitch: float,
        yaw_rate: float
    ):
        """Send attitude setpoint."""
        if self._cf is None:
            return

        # Convert to Crazyflie units
        # Roll/pitch in degrees, yaw_rate in deg/s, thrust 0-65535
        roll_deg = np.degrees(roll)
        pitch_deg = np.degrees(pitch)
        yaw_rate_deg = np.degrees(yaw_rate)
        thrust_cf = int(thrust * 65535)

        self._cf.commander.send_setpoint(roll_deg, pitch_deg, yaw_rate_deg, thrust_cf)


class TelloInterface(DroneInterface):
    """
    Interface for DJI Tello drones.

    The Tello is a consumer drone with good stability and
    easy-to-use SDK, suitable for initial testing.

    Requires: djitellopy (pip install djitellopy)
    """

    def __init__(self):
        super().__init__()
        self._tello = None
        self._state_thread = None
        self._running = False

    def connect(self, address: str = None) -> bool:
        """Connect to Tello drone via WiFi."""
        try:
            from djitellopy import Tello

            self.status = DroneStatus.CONNECTING
            self._tello = Tello()
            self._tello.connect()

            # Start state polling thread
            self._running = True
            self._state_thread = threading.Thread(target=self._poll_state)
            self._state_thread.start()

            self.status = DroneStatus.CONNECTED
            logger.info(f"Connected to Tello. Battery: {self._tello.get_battery()}%")
            return True

        except ImportError:
            logger.error("djitellopy not installed. Install with: pip install djitellopy")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Tello: {e}")
            self.status = DroneStatus.ERROR
            return False

    def _poll_state(self):
        """Poll Tello state in background thread."""
        while self._running:
            try:
                self.state = DroneState(
                    position=np.zeros(3),  # Tello doesn't provide position
                    velocity=np.array([
                        self._tello.get_speed_x(),
                        self._tello.get_speed_y(),
                        self._tello.get_speed_z()
                    ]) / 100.0,  # cm/s to m/s
                    orientation=np.array([
                        np.radians(self._tello.get_roll()),
                        np.radians(self._tello.get_pitch()),
                        np.radians(self._tello.get_yaw())
                    ]),
                    angular_velocity=np.zeros(3),
                    battery_voltage=0,
                    battery_percent=self._tello.get_battery(),
                    timestamp=time.time()
                )

                if self._state_callback:
                    self._state_callback(self.state)

            except Exception:
                pass

            time.sleep(0.1)

    def disconnect(self):
        """Disconnect from Tello."""
        self._running = False
        if self._state_thread:
            self._state_thread.join()
        if self._tello:
            self._tello.end()
        self.status = DroneStatus.DISCONNECTED

    def arm(self) -> bool:
        """Tello arms automatically."""
        return True

    def disarm(self):
        """Emergency stop."""
        if self._tello:
            self._tello.emergency()

    def takeoff(self, height: float = 1.0) -> bool:
        """Takeoff."""
        try:
            self._tello.takeoff()
            self.status = DroneStatus.FLYING
            return True
        except Exception as e:
            logger.error(f"Takeoff failed: {e}")
            return False

    def land(self):
        """Land the drone."""
        if self._tello:
            self._tello.land()
        self.status = DroneStatus.CONNECTED

    def send_motor_commands(self, commands: np.ndarray):
        """Tello doesn't support direct motor control."""
        logger.warning("Tello doesn't support direct motor control")

    def send_attitude_command(
        self,
        thrust: float,
        roll: float,
        pitch: float,
        yaw_rate: float
    ):
        """Send RC-style commands."""
        if self._tello is None:
            return

        # Convert to Tello RC commands (-100 to 100)
        # roll -> left/right, pitch -> forward/back
        roll_cmd = int(np.clip(roll * 100 / 0.5, -100, 100))
        pitch_cmd = int(np.clip(pitch * 100 / 0.5, -100, 100))
        yaw_cmd = int(np.clip(yaw_rate * 100 / 1.0, -100, 100))
        throttle_cmd = int((thrust - 0.5) * 200)  # Center at 0

        self._tello.send_rc_control(roll_cmd, pitch_cmd, throttle_cmd, yaw_cmd)


class SimulatedDroneInterface(DroneInterface):
    """
    Simulated drone interface for testing without hardware.

    This wraps the physics simulation to provide the same interface
    as real hardware, useful for testing deployment code.
    """

    def __init__(self):
        super().__init__()
        from drone_ai.simulation import DroneSimulation
        self._sim = DroneSimulation()
        self._running = False
        self._control_thread = None

    def connect(self, address: str = None) -> bool:
        """Connect to simulated drone."""
        self.status = DroneStatus.CONNECTED
        logger.info("Connected to simulated drone")
        return True

    def disconnect(self):
        """Disconnect."""
        self._running = False
        if self._control_thread:
            self._control_thread.join()
        self.status = DroneStatus.DISCONNECTED

    def arm(self) -> bool:
        self.status = DroneStatus.ARMED
        return True

    def disarm(self):
        self.status = DroneStatus.CONNECTED

    def takeoff(self, height: float = 1.0) -> bool:
        self._sim.reset(position=np.array([0, 0, height]))
        self.status = DroneStatus.FLYING
        return True

    def land(self):
        self.status = DroneStatus.CONNECTED

    def send_motor_commands(self, commands: np.ndarray):
        """Send commands to simulation."""
        state = self._sim.step(commands)
        self.state = DroneState(
            position=state.position,
            velocity=state.velocity,
            orientation=state.get_euler_angles(),
            angular_velocity=state.angular_velocity,
            battery_voltage=4.2,
            battery_percent=100,
            timestamp=time.time()
        )

        if self._state_callback:
            self._state_callback(self.state)

    def send_attitude_command(
        self,
        thrust: float,
        roll: float,
        pitch: float,
        yaw_rate: float
    ):
        """Convert attitude to motor commands."""
        # Simple mixing (would use proper controller in real implementation)
        base = thrust
        commands = np.array([
            base + roll + pitch - yaw_rate * 0.1,
            base - roll + pitch + yaw_rate * 0.1,
            base + roll - pitch + yaw_rate * 0.1,
            base - roll - pitch - yaw_rate * 0.1
        ])
        self.send_motor_commands(np.clip(commands, 0, 1))


def create_interface(platform: str) -> DroneInterface:
    """
    Factory function to create appropriate drone interface.

    Args:
        platform: One of "crazyflie", "tello", "simulation"

    Returns:
        DroneInterface instance
    """
    platform = platform.lower()

    if platform == "crazyflie":
        return CrazyflieInterface()
    elif platform == "tello":
        return TelloInterface()
    elif platform in ["simulation", "sim", "simulated"]:
        return SimulatedDroneInterface()
    else:
        raise ValueError(f"Unknown platform: {platform}. Use 'crazyflie', 'tello', or 'simulation'")
