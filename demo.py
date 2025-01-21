import streamlit as st
import torch
import time
from minesweeper import MinesweeperEnv
from minesweeper_AI import ConvNet
import torch.nn as nn
import torch.nn.functional as F
import io
import requests
import sys


def select_action(state, done_actions, policy_net):
    with torch.no_grad():
        action_values = policy_net(state.unsqueeze(1))
        action_values[0, done_actions] = -float("inf")
        move = torch.argmax(action_values).item()
    return move


def play_a_game(policy_net):
    move_delay = 1
    done_actions = []
    _, state = env.reset()
    state = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)

    # Capture console output using StringIO
    captured_output = io.StringIO()
    original_stdout = sys.stdout  # Store original standard output
    sys.stdout = captured_output

    env.print_player_grid()

    st.write(captured_output.getvalue())
    captured_output.seek(0)

    while True:

        action = select_action(state, done_actions, policy_net)
        observation, _, endstate, won, recursion_list, action = env.step(
            action, done_actions, False
        )
        done_actions.append(int(action))
        done_actions.extend(recursion_list)
        state = torch.tensor(observation, dtype=torch.float, device=device).unsqueeze(0)

        env.print_player_grid()

        st.write(captured_output.getvalue())
        captured_output.seek(0)  # Reset StringIO

        if endstate:
            if won:
                st.write("VICTORY")
            else:
                st.write("DEFEAT")
            break

        time.sleep(move_delay)

    # Restore original standard output
    sys.stdout = original_stdout
    return won


st.title("AI Minesweeper")

xdim, ydim, total_mines = 5, 5, 5
env = MinesweeperEnv(xdim, ydim, total_mines)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

load_release = False


if load_release:

    model_url = "https://github.com/your_username/your_repository_name/releases/download/v1.0/your_model_filename.pth"  # Replace with actual URL

    try:
        response = requests.get(model_url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Save the downloaded model
        with open("downloaded_model.pth", "wb") as f:
            f.write(response.content)

        # Load model from file
        model = torch.load("downloaded_model.pth")
        model.eval()  # Set model to evaluation mode, not sure if neccessary, or if it does anything at all

    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading model: {e}")

else:
    model = ConvNet(xdim, ydim).to(device)
    model.load_state_dict(torch.load("./saved_states/safed_policy_netCNNsmallest.pth"))
    print(type(model))


# Button to start a new game
if st.button("Start New Game"):
    play_a_game(model)
