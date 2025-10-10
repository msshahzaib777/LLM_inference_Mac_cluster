from vlm.environment2d.environment2d import Environment2D
from vlm.model_test import parse_llm_action
import random

env_name = "env4"
create_new_env = False
plot_path = f"/Users/studentone/Documents/LLM_inference/vlm/samples/{env_name}"

default_config = {
        'rectangle_size_range': (4, 30),
        'circle_radius_range': (4, 9),
        'wall_length_range': (20, 30),
        'wall_thickness': 10,
        'allow_diagonal_walls': False,
        'obstacle_margin': 5
    }
env = Environment2D(width=100, height=100, grid_resolution=1, config=default_config)
if create_new_env:
    margin, min_dist = 5, 20
    start = (random.randint(margin, env.width - margin), random.randint(margin, env.height - margin))
    goal = start
    while ((goal[0] - start[0]) ** 2 + (goal[1] - start[1]) ** 2) ** 0.5 < min_dist:
        goal = (random.randint(margin, env.width - margin), random.randint(margin, env.height - margin))
    env.set_start(*start)
    env.set_goal(*goal)

    obstacle_distribution = {
        "rectangle": 0.5,
        "circle": 0,
        "wall": 0.5
    }
    env.generate_obstacles(num_obstacles=5, distribution=obstacle_distribution)
    env.save_environment(f"{plot_path}/{env_name}.pkl")
else:
    env = env.load_environment(f"{plot_path}/{env_name}.pkl")
env.render()

step_num = 0
image_path = env.get_environment_image(f"{plot_path}/{env_name}_{step_num:03d}.png")
step_num = 1
while True:

    assistant_output = input("Assistant response (e.g., explanation + 'Action: east, 3'): ")
    direction, step, explanation = parse_llm_action(assistant_output)

    if direction == "finished" and step == "finished":
        print("Goal reached!")
        break

    if direction and step:
        try:
            image_path = env.take_step(
                direction,
                step_size=step,
                step_num=step_num,
                save_prefix=f"{plot_path}/{env_name}",
                explanation=explanation
            )
        except ValueError as e:
            print(f"Step blocked: {e}")
            continue

        env.render()
        step_num += 1
    else:
        print("Could not parse direction/step. Skipping movement.")
