import random
from vlm.environment2d.environment2d import Environment2D

samples_folder = "/Users/studentone/Documents/LLM_inference/vlm/samples"
env_name = "env_2"
create_new_env = True
plot_path = f"{samples_folder}/{env_name}"

def create_and_save_environment():
    default_config = {
        'rectangle_size_range': (1, 1),
        'circle_radius_range': (1, 1),
        'wall_length_range': (1, 1),
        'wall_thickness': 1,
        'allow_diagonal_walls': False,
        'obstacle_margin': 1
    }
    env = Environment2D(width=10, height=10, grid_resolution=1, config=default_config)
    if create_new_env:
        margin, min_dist = 1, 20
        start = (random.randint(margin, env.width - margin), random.randint(margin, env.height - margin))
        goal = start
        while ((goal[0] - start[0]) ** 2 + (goal[1] - start[1]) ** 2) ** 0.5 < min_dist:
            goal = (random.randint(margin, env.width - margin), random.randint(margin, env.height - margin))
        env.set_start(*start)
        env.set_goal(*goal)

        obstacle_distribution = {
            "rectangle": 1,
            "circle": 0,    
            "wall": 0
        }
        # env.generate_obstacles(num_obstacles=5, distribution=obstacle_distribution)
        env.generate_maze()
        env.save_environment(f"{plot_path}/{env_name}.pkl")

    else:
        env = env.load_environment(f"{plot_path}/{env_name}.pkl")
    image_path = env.get_environment_image(f"{plot_path}/{env_name}.png")
    env.render()

# Run this
create_and_save_environment()
