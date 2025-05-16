# import gym
import argparse
import os
import time

import box_world_env

# from PIL import Image
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Run environment with random selected actions.")
parser.add_argument("--rounds", "-r", metavar="rounds", type=int, help="number of rounds to play (default: 1)", default=1)
parser.add_argument(
    "--steps",
    "-s",
    metavar="steps",
    type=int,
    help="maximum number of steps to be played each round (default: 300)",
    default=300,
)
parser.add_argument("--save", action="store_true", help="Save images of single steps")
parser.add_argument("--gifs", action="store_true", help="Generate Gif files from images")

args = parser.parse_args()
env_name = "boxplot"
n_rounds = args.rounds
n_steps = args.steps
save_images = args.save or args.gifs
generate_gifs = args.gifs

# Creating target directory if images are to be stored
if save_images and not os.path.exists("images"):
    try:
        os.makedirs("images")
    except OSError:
        print("Error: Creating images target directory. ")

ts = time.time()

# env = box_world_env.BoxWorld(12, 4, 2, 2)
env = box_world_env.BoxWorld(6, 2, 1, 1, collect_key=False)

ACTION_LOOKUP = env.unwrapped.get_action_lookup()
print("Created environment: {}".format(env_name))


def print_available_actions():
    """
    Prints all available actions nicely formatted..
    :return:
    """
    available_actions_list = []
    for i in range(len(ACTION_LOOKUP)):
        available_actions_list.append("Key: {} - Action: {}".format(i, ACTION_LOOKUP[i]))
    display_actions = "\n".join(available_actions_list)
    print()
    print("Action out of Range!")
    print("Available Actions:\n{}".format(display_actions))
    print()


for i_episode in range(n_rounds):
    print("Starting new game!")
    observation = env.reset()

    for t in range(n_steps):
        env.render()

        action = input("Select action: ")
        try:
            action = int(action)

            if action not in range(len(ACTION_LOOKUP)):
                raise ValueError

        except ValueError:
            print_available_actions()
            continue

        observation, reward, done, info = env.step(action)
        print(ACTION_LOOKUP[action], reward, done, info)
        print(len(observation), len(observation[0]), len(observation[0][0]))
        if save_images:
            # img = Image.fromarray(env.render(mode="return"), 'RGB')
            # img.save(os.path.join('images', 'observation_{}_{}.png'.format(i_episode, t)))
            img = env.render(mode="rgb_array")
            fig = plt.imshow(img, vmin=0, vmax=255, interpolation="none")
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.savefig(os.path.join("images", "observation_{}_{}.png".format(i_episode, t)))

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            env.render()
            break

    if generate_gifs:
        print("")
        import imageio

        with imageio.get_writer(os.path.join("images", "round_{}.gif".format(i_episode)), mode="I", fps=2) as writer:
            for t in range(n_steps):
                try:
                    filename = os.path.join("images", "observation_{}_{}.png".format(i_episode, t))
                    image = imageio.imread(filename)
                    writer.append_data(image)

                except:
                    pass

env.close()
time.sleep(10)
