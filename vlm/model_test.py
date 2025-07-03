import random
import re
from PIL import Image
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from vlm.environment2d.environment2d import Environment2D

plot_path = "/Users/studentone/Documents/LLM_inference/vlm/samples"
env_name = "env1"

model_path = "mlx-community/Qwen2.5-VL-32B-Instruct-8bit"
# model_path =  "Qwen/Qwen3-32B-MLX-8bit"
# model_path = "mlx-community/Qwen2.5-VL-72B-Instruct-4bit"
# model_path = "mlx-community/Qwen3-8B-bf16"

def parse_llm_action(output):
    output = output.strip()

    # Check for final answer
    if "Final Answer: Finished" in output:
        return "finished", "finished", None

    # Extract explanation if present
    explanation_match = re.search(r"Explanation:\s*(.+?)(?=Action:|$)", output, re.IGNORECASE | re.DOTALL)
    explanation = explanation_match.group(1).strip() if explanation_match else None

    # Extract action
    action_match = re.search(r"Action:\s*([a-zA-Z]+)\s*,\s*(\d+)", output, re.IGNORECASE)
    if action_match:
        direction = action_match.group(1).lower()
        step = int(action_match.group(2))
        return direction, step, explanation

    print("Invalid format: Could not find valid 'Action: <direction>, <step>'")
    return None, None, None

def run_vlm_guided_navigation():
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
    # env.save_environment(f"{plot_path}/pkls/{env_name}.pkl")
    env = env.load_environment(f"{plot_path}/pkls/{env_name}.pkl")
    image_path = env.get_environment_image(f"{plot_path}/{env_name}.png")
    model, processor = load(model_path)
    config = load_config(model_path)

    prompt = '''I am an intelligent path planning agent. My task is to guide the robot from its current position to the goal using visual observations of a 2D environment.
The environment contains:
- Green dot: starting position  
- Red dot: goal  
- Blue dot: current robot position  
- Black rectangles: obstacles  
- Thin blue lines: robot's explored path

At each step:
1. I analyze the visual layout, noting the positions of the robot, goal, and obstacles.
2. I choose the optimal compass direction (e.g., north, southeast) to avoid collisions and reduce distance to the goal.
3. I determine the step size based on visible free space in the selected direction.

I must respond with:
**Action: <direction>, <step size>**

Where:
- <direction> is one of: north, northeast, east, southeast, south, southwest, west, northwest  
- <step size> is a positive integer based on how far the robot can safely move

If the robot has reached the goal, I respond with:
**Final Answer: Finished!**

I always include a brief explanation of my observation and decision before the action.
'''

    prompt2 = '''You are my navigation agent. Given a 2D image showing a robot (blue), goal (red), start (green), obstacles (black), and path (thin blue), guide the robot toward the goal.

Only respond with:
Action: <direction>, <step size>

Direction must be one of: north, northeast, east, southeast, south, southwest, west, northwest.  
Step size is an integer based on visual distance to the goal without hitting obstacles.

Do not add explanations or reasoning. If the goal is reached:
Final Answer: Finished!
    '''
    messages = [
        {"role": "assistant", "content": prompt},
        {"role": "user", "content": f"Here is the current environment. What should be the next move?",
                     "images": [image_path]}
    ]
    for step_num in range(50):
        formatted_prompt = apply_chat_template(
            processor, config, messages, num_images=1
        )
        output = generate(
            model, processor, formatted_prompt, [Image.open(image_path)], verbose=False
        )

        messages.append({
            "role": "assistant",
            "content": output[0].strip()  # clean output if needed
        })

        direction, step = parse_llm_action(output[0])
        print(f"Step {step_num}:", output[0])
        if "finished" in output[0].lower():
            print("Goal reached!")
            break
        try:
            image_path = env.take_step(direction, step_size=step, step_num=step_num, save_prefix=f"{plot_path}/{env_name}")
            messages.append({"role": "user",
                             "content": f"What should be the next move to reach the goal(red dot) while avoiding obstacles? and stop the navigation and respond with **Final Answer: Finished!**",
                             "images": [image_path]})
        except ValueError as e:
            messages.append({"role": "user", "content": f"Previous Step blocked: {e}",
             "images": [image_path]})
        print(messages[-1])

# run_vlm_guided_navigation()