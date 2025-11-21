"""
3D Visualization System for Drone Simulation

This module provides real-time 3D visualization of the drone simulation
using Pygame. Features include:
- 3D drone rendering with orientation
- Flight trajectory trail
- Target position marker
- Ground grid
- Camera controls
- HUD with telemetry data
"""

import numpy as np
import math
from typing import List, Optional, Tuple
from dataclasses import dataclass

try:
    import pygame
    from pygame import gfxdraw
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from drone_ai.simulation import DroneState, PackageState, PackageStatus


@dataclass
class CameraState:
    """Camera configuration for 3D view."""
    distance: float = 5.0  # Distance from target
    azimuth: float = 45.0  # Horizontal angle (degrees)
    elevation: float = 30.0  # Vertical angle (degrees)
    target: np.ndarray = None  # Look-at target

    def __post_init__(self):
        if self.target is None:
            self.target = np.array([0.0, 0.0, 1.0])


class DroneRenderer:
    """
    Real-time 3D drone visualization.

    Controls:
        - Arrow keys: Rotate camera
        - +/-: Zoom in/out
        - R: Reset camera
        - Space: Toggle follow mode
        - ESC: Close window
    """

    # Colors
    BACKGROUND = (20, 20, 30)
    GRID_COLOR = (50, 50, 60)
    DRONE_BODY = (100, 150, 255)
    DRONE_ARM = (80, 80, 80)
    MOTOR_CW = (255, 100, 100)  # Clockwise motors
    MOTOR_CCW = (100, 255, 100)  # Counter-clockwise motors
    TARGET = (255, 200, 50)
    TRAJECTORY = (100, 200, 255)
    TEXT_COLOR = (200, 200, 200)
    GROUND = (40, 40, 50)
    # Delivery task colors
    PACKAGE = (255, 150, 50)       # Orange package
    PACKAGE_ATTACHED = (50, 255, 50)  # Green when attached
    PICKUP_ZONE = (100, 255, 255)  # Cyan pickup zone
    DROPZONE = (255, 100, 255)     # Magenta drop zone
    DROPZONE_SUCCESS = (50, 255, 50)  # Green on successful delivery

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        headless: bool = False
    ):
        """Initialize the renderer."""
        if not PYGAME_AVAILABLE:
            raise ImportError("Pygame is required for visualization. Install with: pip install pygame")

        self.width = width
        self.height = height
        self.headless = headless

        # Initialize pygame
        if headless:
            pygame.init()
            self.screen = pygame.Surface((width, height))
        else:
            pygame.init()
            pygame.display.set_caption("Drone AI - Flight Visualization")
            self.screen = pygame.display.set_mode((width, height))

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)

        # Camera state
        self.camera = CameraState()
        self.follow_mode = True

        # Rendering settings
        self.show_trajectory = True
        self.show_hud = True
        self.max_trajectory_points = 500

    def render(
        self,
        state: DroneState,
        target: np.ndarray,
        trajectory: List[np.ndarray],
        package: Optional[PackageState] = None,
        dropzone_radius: float = 0.3
    ):
        """Render the current state.

        Args:
            state: Current drone state
            target: Target position
            trajectory: List of past positions
            package: Optional package state for delivery task
            dropzone_radius: Radius of the drop zone
        """
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            self._handle_input(event)

        # Handle continuous key presses
        self._handle_continuous_input()

        # Update camera target if following
        if self.follow_mode:
            self.camera.target = state.position.copy()

        # Clear screen
        self.screen.fill(self.BACKGROUND)

        # Render 3D scene
        self._render_ground()

        # Render delivery elements if package exists
        if package is not None:
            self._render_delivery_zones(package, dropzone_radius)
            self._render_package(package, state)

        self._render_trajectory(trajectory)
        self._render_target(target)
        self._render_drone(state)

        # Render HUD
        if self.show_hud:
            self._render_hud(state, target, package)

        # Update display
        if not self.headless:
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS

    def render_to_array(
        self,
        state: DroneState,
        target: np.ndarray,
        trajectory: List[np.ndarray],
        package: Optional[PackageState] = None,
        dropzone_radius: float = 0.3
    ) -> np.ndarray:
        """Render and return as numpy array."""
        self.render(state, target, trajectory, package, dropzone_radius)
        return pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)

    def _handle_input(self, event):
        """Handle single key press events."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
            elif event.key == pygame.K_r:
                self.camera = CameraState()
            elif event.key == pygame.K_SPACE:
                self.follow_mode = not self.follow_mode
            elif event.key == pygame.K_t:
                self.show_trajectory = not self.show_trajectory
            elif event.key == pygame.K_h:
                self.show_hud = not self.show_hud

    def _handle_continuous_input(self):
        """Handle continuous key presses."""
        keys = pygame.key.get_pressed()

        # Camera rotation
        rotation_speed = 2.0
        if keys[pygame.K_LEFT]:
            self.camera.azimuth -= rotation_speed
        if keys[pygame.K_RIGHT]:
            self.camera.azimuth += rotation_speed
        if keys[pygame.K_UP]:
            self.camera.elevation = min(89, self.camera.elevation + rotation_speed)
        if keys[pygame.K_DOWN]:
            self.camera.elevation = max(-89, self.camera.elevation - rotation_speed)

        # Zoom
        zoom_speed = 0.1
        if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS]:
            self.camera.distance = max(1, self.camera.distance - zoom_speed)
        if keys[pygame.K_MINUS]:
            self.camera.distance = min(20, self.camera.distance + zoom_speed)

    def _world_to_screen(self, point: np.ndarray) -> Tuple[int, int]:
        """Project 3D world point to 2D screen coordinates."""
        # Camera position
        az = math.radians(self.camera.azimuth)
        el = math.radians(self.camera.elevation)

        cam_x = self.camera.target[0] + self.camera.distance * math.cos(el) * math.sin(az)
        cam_y = self.camera.target[1] + self.camera.distance * math.cos(el) * math.cos(az)
        cam_z = self.camera.target[2] + self.camera.distance * math.sin(el)
        cam_pos = np.array([cam_x, cam_y, cam_z])

        # View direction
        forward = self.camera.target - cam_pos
        forward = forward / np.linalg.norm(forward)

        # Right and up vectors
        world_up = np.array([0, 0, 1])
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        # Project point
        rel = point - cam_pos
        x = np.dot(rel, right)
        y = np.dot(rel, up)
        z = np.dot(rel, forward)

        if z <= 0.1:
            return None  # Behind camera

        # Perspective projection
        fov = 60
        scale = self.height / (2 * math.tan(math.radians(fov / 2)))

        screen_x = int(self.width / 2 + x * scale / z)
        screen_y = int(self.height / 2 - y * scale / z)

        return (screen_x, screen_y)

    def _render_ground(self):
        """Render ground grid."""
        grid_size = 10
        grid_spacing = 1.0

        for i in range(-grid_size, grid_size + 1):
            # Lines parallel to X
            start = self._world_to_screen(np.array([i * grid_spacing, -grid_size * grid_spacing, 0]))
            end = self._world_to_screen(np.array([i * grid_spacing, grid_size * grid_spacing, 0]))
            if start and end:
                pygame.draw.line(self.screen, self.GRID_COLOR, start, end, 1)

            # Lines parallel to Y
            start = self._world_to_screen(np.array([-grid_size * grid_spacing, i * grid_spacing, 0]))
            end = self._world_to_screen(np.array([grid_size * grid_spacing, i * grid_spacing, 0]))
            if start and end:
                pygame.draw.line(self.screen, self.GRID_COLOR, start, end, 1)

    def _render_trajectory(self, trajectory: List[np.ndarray]):
        """Render flight trajectory."""
        if not self.show_trajectory or len(trajectory) < 2:
            return

        # Limit trajectory length
        traj = trajectory[-self.max_trajectory_points:]

        points = []
        for pos in traj:
            screen_pos = self._world_to_screen(pos)
            if screen_pos:
                points.append(screen_pos)

        if len(points) >= 2:
            pygame.draw.lines(self.screen, self.TRAJECTORY, False, points, 2)

    def _render_target(self, target: np.ndarray):
        """Render target position."""
        screen_pos = self._world_to_screen(target)
        if screen_pos:
            # Draw target marker
            pygame.draw.circle(self.screen, self.TARGET, screen_pos, 10, 2)
            pygame.draw.line(self.screen, self.TARGET,
                           (screen_pos[0] - 15, screen_pos[1]),
                           (screen_pos[0] + 15, screen_pos[1]), 2)
            pygame.draw.line(self.screen, self.TARGET,
                           (screen_pos[0], screen_pos[1] - 15),
                           (screen_pos[0], screen_pos[1] + 15), 2)

    def _render_drone(self, state: DroneState):
        """Render the drone."""
        pos = state.position
        R = state.get_rotation_matrix()

        # Drone dimensions
        arm_length = 0.2
        body_size = 0.1

        # Motor positions in body frame (X-configuration)
        motor_positions_body = [
            np.array([arm_length * 0.707, arm_length * 0.707, 0]),   # Front-left
            np.array([arm_length * 0.707, -arm_length * 0.707, 0]),  # Front-right
            np.array([-arm_length * 0.707, arm_length * 0.707, 0]),  # Rear-left
            np.array([-arm_length * 0.707, -arm_length * 0.707, 0]), # Rear-right
        ]

        # Transform to world frame
        motor_positions = [pos + R @ mp for mp in motor_positions_body]

        # Draw arms
        center_screen = self._world_to_screen(pos)
        if center_screen:
            for i, motor_pos in enumerate(motor_positions):
                motor_screen = self._world_to_screen(motor_pos)
                if motor_screen:
                    pygame.draw.line(self.screen, self.DRONE_ARM, center_screen, motor_screen, 3)

            # Draw body
            pygame.draw.circle(self.screen, self.DRONE_BODY, center_screen, 8)

            # Draw motors
            motor_colors = [self.MOTOR_CW, self.MOTOR_CCW, self.MOTOR_CCW, self.MOTOR_CW]
            for motor_pos, color in zip(motor_positions, motor_colors):
                motor_screen = self._world_to_screen(motor_pos)
                if motor_screen:
                    pygame.draw.circle(self.screen, color, motor_screen, 5)

            # Draw direction indicator (forward)
            forward_body = np.array([body_size * 1.5, 0, 0])
            forward_world = pos + R @ forward_body
            forward_screen = self._world_to_screen(forward_world)
            if forward_screen:
                pygame.draw.line(self.screen, (255, 255, 100), center_screen, forward_screen, 2)

    def _render_delivery_zones(self, package: PackageState, dropzone_radius: float):
        """Render pickup and drop zones for delivery task."""
        # Pickup zone (cyan circle on ground)
        pickup_pos = package.pickup_position.copy()
        pickup_pos[2] = 0.01  # Slightly above ground to be visible

        # Draw pickup zone as a circle
        n_segments = 16
        pickup_points = []
        for i in range(n_segments + 1):
            angle = 2 * np.pi * i / n_segments
            point = pickup_pos + np.array([
                0.2 * np.cos(angle),
                0.2 * np.sin(angle),
                0
            ])
            screen_point = self._world_to_screen(point)
            if screen_point:
                pickup_points.append(screen_point)

        if len(pickup_points) >= 2:
            # Draw filled if package is waiting, outline otherwise
            if package.status == PackageStatus.WAITING:
                pygame.draw.polygon(self.screen, self.PICKUP_ZONE, pickup_points)
            else:
                pygame.draw.lines(self.screen, self.PICKUP_ZONE, True, pickup_points, 2)

        # Drop zone (magenta/green circle on ground)
        dropzone_pos = package.dropzone_position.copy()
        dropzone_pos[2] = 0.01

        # Determine drop zone color based on status
        if package.status == PackageStatus.DELIVERED:
            zone_color = self.DROPZONE_SUCCESS
        else:
            zone_color = self.DROPZONE

        # Draw drop zone circle
        dropzone_points = []
        for i in range(n_segments + 1):
            angle = 2 * np.pi * i / n_segments
            point = dropzone_pos + np.array([
                dropzone_radius * np.cos(angle),
                dropzone_radius * np.sin(angle),
                0
            ])
            screen_point = self._world_to_screen(point)
            if screen_point:
                dropzone_points.append(screen_point)

        if len(dropzone_points) >= 2:
            pygame.draw.lines(self.screen, zone_color, True, dropzone_points, 3)

            # Draw inner target
            inner_points = []
            for i in range(n_segments + 1):
                angle = 2 * np.pi * i / n_segments
                point = dropzone_pos + np.array([
                    dropzone_radius * 0.3 * np.cos(angle),
                    dropzone_radius * 0.3 * np.sin(angle),
                    0
                ])
                screen_point = self._world_to_screen(point)
                if screen_point:
                    inner_points.append(screen_point)

            if len(inner_points) >= 2:
                pygame.draw.lines(self.screen, zone_color, True, inner_points, 2)

    def _render_package(self, package: PackageState, drone_state: DroneState):
        """Render the package."""
        # Determine package position and color based on status
        if package.status == PackageStatus.WAITING:
            pkg_pos = package.pickup_position.copy()
            pkg_pos[2] = 0.05  # Slightly above ground
            color = self.PACKAGE
        elif package.status == PackageStatus.ATTACHED:
            pkg_pos = drone_state.position.copy()
            pkg_pos[2] -= 0.1  # Below drone
            color = self.PACKAGE_ATTACHED
        elif package.status == PackageStatus.DROPPING:
            pkg_pos = package.position.copy()
            color = self.PACKAGE
        elif package.status in [PackageStatus.DELIVERED, PackageStatus.MISSED]:
            pkg_pos = package.position.copy()
            color = self.DROPZONE_SUCCESS if package.status == PackageStatus.DELIVERED else (255, 50, 50)
        else:
            return

        screen_pos = self._world_to_screen(pkg_pos)
        if screen_pos:
            # Draw package as a small square
            size = 8
            pygame.draw.rect(self.screen, color,
                           (screen_pos[0] - size//2, screen_pos[1] - size//2, size, size))
            pygame.draw.rect(self.screen, (255, 255, 255),
                           (screen_pos[0] - size//2, screen_pos[1] - size//2, size, size), 1)

    def _render_hud(self, state: DroneState, target: np.ndarray, package: Optional[PackageState] = None):
        """Render heads-up display."""
        # Background panel - taller if showing package info
        panel_width = 200
        panel_height = 230 if package is not None else 180
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.fill((0, 0, 0, 150))
        self.screen.blit(panel_surface, (10, 10))

        # Title
        title = self.font_large.render("Drone Telemetry", True, self.TEXT_COLOR)
        self.screen.blit(title, (20, 15))

        # Position
        pos_text = f"Pos: ({state.position[0]:.2f}, {state.position[1]:.2f}, {state.position[2]:.2f})"
        text = self.font.render(pos_text, True, self.TEXT_COLOR)
        self.screen.blit(text, (20, 50))

        # Velocity
        vel = np.linalg.norm(state.velocity)
        vel_text = f"Vel: {vel:.2f} m/s"
        text = self.font.render(vel_text, True, self.TEXT_COLOR)
        self.screen.blit(text, (20, 75))

        # Euler angles (degrees)
        euler = state.get_euler_angles() * 180 / np.pi
        euler_text = f"R/P/Y: {euler[0]:.1f} / {euler[1]:.1f} / {euler[2]:.1f}"
        text = self.font.render(euler_text, True, self.TEXT_COLOR)
        self.screen.blit(text, (20, 100))

        # Distance to target
        dist = np.linalg.norm(state.position - target)
        dist_text = f"Target dist: {dist:.2f} m"
        text = self.font.render(dist_text, True, self.TEXT_COLOR)
        self.screen.blit(text, (20, 125))

        # Motor speeds
        motor_text = f"Motors: {state.motor_speeds[0]:.0f} rpm"
        text = self.font.render(motor_text, True, self.TEXT_COLOR)
        self.screen.blit(text, (20, 150))

        # Package status (if delivery task)
        if package is not None:
            status_colors = {
                PackageStatus.WAITING: self.PICKUP_ZONE,
                PackageStatus.ATTACHED: self.PACKAGE_ATTACHED,
                PackageStatus.DROPPING: self.PACKAGE,
                PackageStatus.DELIVERED: self.DROPZONE_SUCCESS,
                PackageStatus.MISSED: (255, 50, 50)
            }
            status_color = status_colors.get(package.status, self.TEXT_COLOR)
            pkg_text = f"Package: {package.status.value}"
            text = self.font.render(pkg_text, True, status_color)
            self.screen.blit(text, (20, 175))

            # Distance to drop zone
            dropzone_dist = np.linalg.norm(state.position[:2] - package.dropzone_position[:2])
            dz_text = f"Dropzone: {dropzone_dist:.2f} m"
            text = self.font.render(dz_text, True, self.DROPZONE)
            self.screen.blit(text, (20, 200))

        # Controls help (bottom of screen)
        help_text = "Arrow keys: rotate | +/-: zoom | Space: follow | T: trajectory | H: HUD"
        text = self.font.render(help_text, True, (100, 100, 100))
        self.screen.blit(text, (self.width // 2 - text.get_width() // 2, self.height - 25))

    def close(self):
        """Clean up resources."""
        pygame.quit()


def demo():
    """Run a demo visualization with a simple trajectory."""
    from drone_ai.simulation import DroneSimulation

    sim = DroneSimulation()
    renderer = DroneRenderer()

    state = sim.reset(position=np.array([0, 0, 1]))
    target = np.array([2, 2, 1.5])
    trajectory = [state.position.copy()]

    # Simple PD controller for demo
    running = True
    step = 0

    while running:
        # Check for quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Simple control towards target
        pos_error = target - state.position
        hover_action = sim.compute_hover_action()

        # Add small corrections based on position error
        action = hover_action.copy()
        action += np.array([
            pos_error[0] * 0.01,
            -pos_error[0] * 0.01,
            pos_error[1] * 0.01,
            -pos_error[1] * 0.01
        ])
        action = np.clip(action, 0, 1)

        state = sim.step(action)
        trajectory.append(state.position.copy())

        renderer.render(state, target, trajectory)

        step += 1
        if step > 10000:
            running = False

    renderer.close()


if __name__ == "__main__":
    demo()
