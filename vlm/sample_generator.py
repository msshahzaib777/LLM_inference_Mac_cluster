from vlm.environment2d.environment2d import Environment2D
from vlm.model_test import parse_llm_action
import random

def get_multiline_input(prompt="Assistant response (end with 'END' on a new line):"):
    print(prompt)
    lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        lines.append(line)
    return "\n".join(lines)

env_name = "env4"
create_new_env = False
plot_path = f"/Users/studentone/Documents/LLM_inference/vlm/samples/{env_name}"
# Load the model
model_path = "mlx-community/Qwen2.5-VL-32B-Instruct-8bit"
# model_path = "mlx-community/Qwen2.5-VL-72B-Instruct-4bit"

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
# model, processor = load(model_path)
# config = load_config(model_path)

prompt = '''I am an intelligent path planning agent. My task is to guide the robot from its current position to the goal using visual observations of a 2D environment.

The environment contains:
- Green dot: starting position  
- Red dot: goal  
- Blue dot: current robot position  
- Black rectangles: obstacles  
- Thin blue lines: robot's explored path

At each step:
1. I analyze the visual layout, noting the positions of the robot, goal, and obstacles.
2. I identify the best direction to move toward the goal while avoiding obstacles.
3. I determine a safe step size based on the available free space in that direction.

In my response, I must:
- First explain my reasoning based on the robot's current position, obstacles, and goal.
- Then respond with the next action in this exact format:

**Action: <direction>, <step size>**

Where:
- <direction> is one of: north, northeast, east, southeast, south, southwest, west, northwest  
- <step size> is a positive integer representing how far the robot can safely move in the chosen direction.

If the robot has reached the goal, I must respond only with:
**Final Answer: Finished!**

I must not include any extra text outside the explanation and the action line.'''

messages = [{
        "role": "Assistant",
        "content": [prompt]
    }]
step_num = 0
image_path = env.get_environment_image(f"{plot_path}/{env_name}_{step_num:03d}.png")
step_num = 1
while True:

    print(f"Step {step_num} image saved at {image_path}")
    messages.append({
        "role": "user",
        "content": "What should be the next move to reach the goal(red dot) while avoiding obstacles? and stop the navigation and respond with **Final Answer: Finished!**",
        "images": [image_path]
    })

    assistant_output = get_multiline_input("Assistant response (e.g., explanation + 'Action: east, 3'): ")
    messages.append({
        "role": "assistant",
        "content": assistant_output
    })

    direction, step, explanation = parse_llm_action(assistant_output)

    if direction == "finished" and step == "finished":
        print("Goal reached!")
        break

    if direction and step:
        try:
            print(f"Moving {step} steps is {direction}")
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

# Optional: Save the few-shot messages for reuse
import json
with open(f"samples/{env_name}/{env_name}.json", "w") as f:
    json.dump(messages, f, indent=2)
