import os
import matplotlib.pyplot as plt
import numpy as np
import random, pickle
from math import floor
from collections import deque
from shapely.geometry import LineString, Point, box
from vlm.environment2d.obstacles import RectangleObstacle, WallObstacle, CircleObstacle

class Environment2D:
    direction_angles = {
        'north': 90,
        'northeast': 45,
        'east': 0,
        'southeast': 315,
        'south': 270,
        'southwest': 225,
        'west': 180,
        'northwest': 135
    }
    trajectory_log = []
    def __init__(self, width=100, height=100, grid_resolution=1, config=None):
        self.width = width
        self.height = height
        self.grid_res = grid_resolution
        self.start = None
        self.goal = None
        self.robot = None
        self.obstacles = []
        self.grid_width = int(width // grid_resolution)
        self.grid_height = int(height // grid_resolution)
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        self.trajectory_log = []
        # Default configuration if none is provided
        default_config = {
            'rectangle_size_range': (6, 15),
            'circle_radius_range': (4, 7),
            'wall_length_range': (10, 20),
            'wall_thickness': 2,
            'allow_diagonal_walls': False,
            'obstacle_margin': 0
        }
        self.config = default_config if config is None else {**default_config, **config}

    def set_start(self, x, y):
        self.start = (x, y)

    def set_goal(self, x, y):
        self.goal = (x, y)

    def set_robot(self, x, y):
        self.robot = (x, y)

    def _coord_to_grid(self, x, y):
        return int(y // self.grid_res), int(x // self.grid_res)

    def _mark_obstacle_on_grid(self, x, y, w, h):
        i1, j1 = self._coord_to_grid(x, y)
        i2, j2 = self._coord_to_grid(x + w, y + h)
        self.grid[i1:i2+1, j1:j2+1] = 1

    def _is_area_free(self, x, y, w, h, margin):
        i1, j1 = self._coord_to_grid(x - margin, y - margin)
        i2, j2 = self._coord_to_grid(x + w + margin, y + h + margin)
        i1, i2 = max(0, i1), min(self.grid_height - 1, i2)
        j1, j2 = max(0, j1), min(self.grid_width - 1, j2)
        return np.all(self.grid[i1:i2 + 1, j1:j2 + 1] == 0)

    def _is_path_possible(self):
        visited = np.zeros_like(self.grid, dtype=bool)
        start_i, start_j = self._coord_to_grid(*self.start)
        goal_i, goal_j = self._coord_to_grid(*self.goal)
        queue = deque([(start_i, start_j)])

        while queue:
            i, j = queue.popleft()
            if (i, j) == (goal_i, goal_j):
                return True
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (-1, -1), (-1, 1), (1, -1)]:
                ni, nj = i+di, j+dj
                if 0 <= ni < self.grid_height and 0 <= nj < self.grid_width:
                    if self.grid[ni, nj] == 0 and not visited[ni, nj]:
                        visited[ni, nj] = True
                        queue.append((ni, nj))
        return False

    def _clear_obstacle_on_grid(self, x, y, w, h):
        i1, j1 = self._coord_to_grid(x, y)
        i2, j2 = self._coord_to_grid(x + w, y + h)
        self.grid[i1:i2 + 1, j1:j2 + 1] = 0

    def generate_obstacles(self, num_obstacles, distribution, max_retries=50):
        assert abs(sum(distribution.values()) - 1.0) < 1e-6, "Distribution must sum to 1.0"
        retry = 0

        while retry < max_retries:
            self.obstacles.clear()
            self.grid.fill(0)

            if self.start: self._mark_obstacle_on_grid(*self.start, 2, 2)
            if self.goal: self._mark_obstacle_on_grid(*self.goal, 2, 2)

            counts = {k: floor(v * num_obstacles) for k, v in distribution.items()}
            leftover = num_obstacles - sum(counts.values())
            for k in list(distribution.keys()):
                if leftover > 0:
                    counts[k] += 1
                    leftover -= 1
            success = True
            for shape, count in counts.items():
                for i in range(count):
                    for j in range(100):
                        # Generate random x, y with buffer to allow for max object size (20x20)
                        x = random.randint(0, self.width - 20)
                        y = random.randint(0, self.height - 20)

                        if shape == "rectangle":
                            w = random.randint(*self.config['rectangle_size_range'])
                            h = random.randint(*self.config['rectangle_size_range'])
                            if x + w > self.width or y + h > self.height:
                                continue
                            if not self._is_area_free(x, y, w, h, self.config['obstacle_margin']):
                                continue
                            obstacle = RectangleObstacle(x, y, w, h)
                            self.obstacles.append(obstacle)
                            self._mark_obstacle_on_grid(x, y, w, h)
                        elif shape == "circle":
                            r = random.randint(*self.config['circle_radius_range'])
                            if x - r < 0 or x + r > self.width or y - r < 0 or y + r > self.height:
                                continue
                            if not self._is_area_free(x - r, y - r, 2 * r, 2 * r, self.config['obstacle_margin']):
                                continue
                            obstacle = CircleObstacle(x, y, r)
                            self.obstacles.append(obstacle)
                            self._mark_obstacle_on_grid(x - r, y - r, 2 * r, 2 * r)
                        elif shape == "wall":
                            thickness = self.config['wall_thickness']
                            if self.config['allow_diagonal_walls']:
                                dx = random.randint(*self.config['wall_length_range'])
                                dy = random.randint(*self.config['wall_length_range'])
                                x2 = x + dx
                                y2 = y + dy
                                if not (0 <= min(x, x2) and max(x, x2) <= self.width and 0 <= min(y, y2) and max(y,
                                                                                                                 y2) <= self.height):
                                    continue
                            else:
                                if random.choice([True, False]):  # horizontal
                                    dx = random.randint(*self.config['wall_length_range'])
                                    dy = 0
                                else:  # vertical
                                    dx = 0
                                    dy = random.randint(*self.config['wall_length_range'])
                                x2 = x + dx
                                y2 = y + dy
                            min_x, min_y = min(x, x2), min(y, y2)
                            w, h = abs(x2 - x) or thickness, abs(y2 - y) or thickness
                            if not self._is_area_free(min_x, min_y, w or 2, h or 2, self.config['obstacle_margin']):
                                continue
                            obstacle = WallObstacle(x, y, x2, y2)
                            self.obstacles.append(obstacle)

                        # if not self._is_path_possible():
                        #     self.obstacles.pop()
                        #     if shape == "rectangle":
                        #         self._clear_obstacle_on_grid(x, y, w, h)
                        #     elif shape == "circle":
                        #         self._clear_obstacle_on_grid(x - r, y - r, 2 * r, 2 * r)
                        #     elif shape == "wall":
                        #         wall_thickness = self.config['wall_thickness']
                        #         min_x, min_y = min(x, x2), min(y, y2)
                        #         wall_w = abs(x2 - x) or wall_thickness
                        #         wall_h = abs(y2 - y) or wall_thickness
                        #         self._clear_obstacle_on_grid(min_x, min_y, wall_w, wall_h)
                        # else:
                        break
            if success:
                print("Valid environment generated.")
                return True
            else:
                retry += 1
                print(f"Retrying... ({retry})")

        raise RuntimeError("Failed to generate a valid environment after max retries")

    # Utility functions

    def _line_intersects_rect(x1, y1, x2, y2, rx, ry, rw, rh):
        return LineString([(x1, y1), (x2, y2)]).intersects(box(rx, ry, rx + rw, ry + rh))

    def _line_intersects_circle(x1, y1, x2, y2, cx, cy, r):
        return LineString([(x1, y1), (x2, y2)]).distance(Point(cx, cy)) <= r

    def _line_intersects_line(x1, y1, x2, y2, x3, y3, x4, y4):
        return LineString([(x1, y1), (x2, y2)]).intersects(LineString([(x3, y3), (x4, y4)]))

    def _is_path_clear(self, robot_pos, new_pos, obstacles, robot_buffer=1):
        path = LineString([robot_pos, new_pos]).buffer(robot_buffer)

        for obs in obstacles:
            if isinstance(obs, RectangleObstacle):
                if path.intersects(box(obs.x, obs.y, obs.x + obs.width, obs.y + obs.height)):
                    return False
            elif isinstance(obs, CircleObstacle):
                if path.intersects(Point(obs.x, obs.y).buffer(obs.radius)):
                    return False
            elif isinstance(obs, WallObstacle):
                if path.intersects(obs.as_polygon(thickness=2)):
                    return False
        return True

    def move_robot(self, direction, step=1):
        if self.robot is None:
            self.robot = self.start
            self.trajectory_log.append({
                "step": 0,
                "direction": None,
                "step_size": 0,
                "position": self.robot,
                "explanation": None
            })

        if direction not in self.direction_angles:
            raise ValueError(f"Invalid direction '{direction}'")

        angle = np.deg2rad(self.direction_angles[direction])
        dx = step * np.cos(angle)
        dy = step * np.sin(angle)

        new_x = self.robot[0] + dx
        new_y = self.robot[1] + dy

        if not (0 <= new_x <= self.width and 0 <= new_y <= self.height):
            raise ValueError("Blocked: Out of bounds")

        if self._is_path_clear(self.robot, (new_x, new_y), self.obstacles, robot_buffer=2):
            self.robot = (new_x, new_y)
            return True
        else:
             raise ValueError("Blocked: Collision with obstacle")

    def render(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)

        if self.start:
            ax.add_patch(plt.Circle(self.start, radius=2, color='green'))

        if self.goal:
            ax.add_patch(plt.Circle(self.goal, radius=2, color='red'))

        if self.robot:
            ax.add_patch(plt.Circle(self.robot, radius=2, color='blue'))

        for obs in self.obstacles:
            obs.draw(ax)

        # Draw robot path
        if len(self.trajectory_log) > 1:
            points = [step["position"] for step in self.trajectory_log]
            xs, ys = zip(*points)
            ax.plot(xs, ys, color='lightblue', linewidth=1.5)

        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')
        plt.show()

    def take_step(self, direction, step_size, step_num, save_prefix='step', explanation=None):

        robot_movement = self.move_robot(direction, step=step_size)
        if robot_movement:
            self.trajectory_log.append({
                "step": step_num,
                "direction": direction,
                "step_size": step_size,
                "position": self.robot,
                "explanation": explanation
            })
        plot_path = self.save_plot(f"{save_prefix}_{step_num:03d}.png")
        return plot_path

    def simulate_random_steps(self, num_steps=10, save_prefix='step', step_size=1):
        directions = list(self.direction_angles.keys())
        for step_num in range(num_steps):
            # direction = random.choice(directions)
            direction = directions[1]
            self.take_step(self, direction, step_size, step_num, save_prefix=save_prefix)

    def get_environment_image(self, filename='env_step.png'):
        self.save_plot(filename)
        return filename

    def save_plot(self, filename):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)

        if self.start:
            ax.add_patch(plt.Circle(self.start, radius=2, color='green'))

        if self.goal:
            ax.add_patch(plt.Circle(self.goal, radius=2, color='red'))

        if self.robot:
            ax.add_patch(plt.Circle(self.robot, radius=2, color='blue'))

        for obs in self.obstacles:
            obs.draw(ax)

        # Draw robot path
        if len(self.trajectory_log) > 1:
            points = [step["position"] for step in self.trajectory_log]
            xs, ys = zip(*points)
            ax.plot(xs, ys, color='lightblue', linewidth=1.5)

        ax.set_aspect('equal', adjustable='box')
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        return filename

    # Persistence methods
    def save_environment(self, filename="environment.pkl"):
        # Before saving
        directory = os.path.dirname(filename)
        os.makedirs(directory, exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def load_environment(self, filename="environment.pkl"):
        with open(filename, 'rb') as f:
            return pickle.load(f)

