import random

import matplotlib.pyplot as plt
from vlm.environment2d.environment2d import Environment2D


def create_and_save_environment(filename="env_output.png"):
    default_config = {
        'rectangle_size_range': (4, 30),
        'circle_radius_range': (4, 9),
        'wall_length_range': (20, 30),
        'wall_thickness': 10,
        'allow_diagonal_walls': False,
        'obstacle_margin': 5
    }
    env = Environment2D(width=100, height=100, grid_resolution=1, config=default_config)
    margin, min_dist = 5, 20
    start = (random.randint(margin, env.width - margin), random.randint(margin, env.height - margin))
    goal = start
    while abs(goal[0] - start[0]) < min_dist and abs(goal[1] - start[1]) < min_dist:
        goal = (random.randint(margin, env.width - margin), random.randint(margin, env.height - margin))

    env.set_start(*start)
    env.set_goal(*goal)
    env.set_robot(*start)

    obstacle_distribution = {
        "rectangle": 0.5,
        "circle": 0,
        "wall": 0.5
    }

    env.generate_obstacles(num_obstacles=5, distribution=obstacle_distribution)

    # Render and save to file
    fig, ax = plt.subplots()
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)

    if env.start:
        ax.add_patch(plt.Circle(env.start, radius=2, color='green'))

    if env.goal:
        ax.add_patch(plt.Circle(env.goal, radius=2, color='red'))

    directions = list(env.direction_angles.keys())

    for step_num in range(0, 10):
        direction = directions[1]
        env.take_step(direction, 5, step_num, save_prefix="Scenario_1" )
    print(f"Environment saved to {filename}")

# Run this
create_and_save_environment()
