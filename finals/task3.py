import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl
import pickle
import torch.nn as nn
import torch
import joblib  # For saving and loading models

# /* Deprecated Codes */
# Set the model type: "neural_network" or "random_forest"
# neural_network_or_random_forest = "neural_network"  # Change to "random_forest" to use Random Forest models
# /* End of Deprecated Codes */

# Free combination of the following models to compare the performance
# Note that "random_forest_smooth" means random forest with smooth filtering
# md stands for max_depth
test_models = ["neural_network","random_forest_md2","random_forest_md10"] # For task 3.1 and 3.2
#test_models = ["neural_network","random_forest_md10", "random_forest_smooth_md10"] # For task 3.3

# MLP Model Definition
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 128),  # Input layer to hidden layer (4 inputs: time + goal positions)
            nn.ReLU(),
            nn.Linear(128, 1)   # Hidden layer to output layer
        )

    def forward(self, x):
        return self.model(x)

def main():
    # Load the saved data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, 'data.pkl')  # Replace with your actual filename
    if not os.path.isfile(filename):
        print(f"Error: File {filename} not found in {script_dir}")
        return
    else:
        with open(filename, 'rb') as f:
            data = pickle.load(f)

    # Extract data
    time_array = np.array(data['time'])            # Shape: (N,)
    # Optional: Normalize time data for better performance
    time_array = (time_array - time_array.min()) / (time_array.max() - time_array.min())
        # Generate a new goal position
    goal_position_bounds = {
        'x': (0.6, 0.8),
        'y': (-0.1, 0.1),
        'z': (0.12, 0.12)
    }
    # Create a set of goal positions
    number_of_goal_positions_to_test = 10
    goal_positions = []
    for i in range(number_of_goal_positions_to_test):
        goal_positions.append([
            np.random.uniform(*goal_position_bounds['x']),
            np.random.uniform(*goal_position_bounds['y']),
            np.random.uniform(*goal_position_bounds['z'])
        ])

    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Configuration for the simulation
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=root_dir)  # Initialize simulation interface

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, root_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    controlled_frame_name = "panda_link8"
    init_joint_angles = sim.GetInitMotorAngles()
    init_cartesian_pos, init_R = dyn_model.ComputeFK(init_joint_angles, controlled_frame_name)
    #print(f"Initial joint angles: {init_joint_angles}")

    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors

    # PD controller gains
    kp = 1000  # Proportional gain
    kd = 100   # Derivative gain

    # Get joint velocity limits
    joint_vel_limits = sim.GetBotJointsVelLimit()

    time_step = sim.GetTimeStep()
    # Generate test time array
    test_time_array = np.arange(time_array.min(), time_array.max(), time_step)

    for goal_position in goal_positions:
        print("Testing new goal position------------------------------------")
        print(f"Goal position: {goal_position}")
        # Data collection
        q_mes_all_for_all_models = []
        q_des_all_for_all_models = []
        q_pred_all_for_all_models = []
        qd_pred_all_for_all_models = []
        tau_cmd_all_for_all_models = []
        for test_model in test_models:
            neural_network_or_random_forest = test_model
            #print(f"Testing new goal position for {test_model}-------------------")
            
            # Load all the models in a list
            models = []
            if neural_network_or_random_forest == "neural_network":
                for joint_idx in range(num_joints):
                    # Instantiate the model
                    model = MLP()
                    # Load the saved model
                    model_filename = os.path.join(script_dir, f'neuralq{joint_idx+1}.pt')
                    model.load_state_dict(torch.load(model_filename))
                    model.eval()
                    models.append(model)
            # Deprecated Codes
            # elif neural_network_or_random_forest == "random_forest" or \
            #         neural_network_or_random_forest == "random_forest_smooth":
            #     for joint_idx in range(num_joints):
            #         # Load the saved Random Forest model
            #         model_filename = os.path.join(script_dir, f'rf_joint{joint_idx+1}.joblib')
            #         model = joblib.load(model_filename)
            #         models.append(model)
            # Extract the max_depth from the model name and load model
            elif "random_forest" in neural_network_or_random_forest \
                        and "md" in neural_network_or_random_forest:
                max_depth = int(neural_network_or_random_forest.split("_md")[1])
                for joint_idx in range(num_joints):
                    # Load the saved Random Forest model
                    model_filename = os.path.join(script_dir, f'rf_joint{joint_idx+1}_md{max_depth}.joblib')
                    model = joblib.load(model_filename)
                    models.append(model)
            else:
                print("Invalid model type specified. Please set neural_network_or_random_forest to 'neural_network' or 'random_forest'")
                return

            # Initialize the simulation
            sim.ResetPose()
            current_time = 0  # Initialize current time

            # Create test input features
            test_goal_positions = np.tile(goal_position, (len(test_time_array), 1))  # Shape: (num_points, 3)
            test_input = np.hstack((test_time_array.reshape(-1, 1), test_goal_positions))  # Shape: (num_points, 4)

            # Predict joint positions for the new goal position
            predicted_joint_positions_over_time = np.zeros((len(test_time_array), 7))  # Shape: (num_points, 7)

            for joint_idx in range(num_joints):
                if neural_network_or_random_forest == "neural_network":
                    # Prepare the test input
                    test_input_tensor = torch.from_numpy(test_input).float()  # Shape: (num_points, 4)

                    # Predict joint positions using the neural network
                    with torch.no_grad():
                        predictions = models[joint_idx](test_input_tensor).numpy().flatten()  # Shape: (num_points,)
                elif "random_forest" in neural_network_or_random_forest:
                    # Predict joint positions using the Random Forest
                    predictions = models[joint_idx].predict(test_input)  # Shape: (num_points,)

                # Store the predicted joint positions
                predicted_joint_positions_over_time[:, joint_idx] = predictions
            
            # Implement smooth filtering for Random Forest models using exponential moving average filter
            # if random_forest_smooth in the model name
            if "random_forest_smooth" in neural_network_or_random_forest:
                # Initialize the filtered joint positions
                filtered_predicted_joint_positions_over_time = np.zeros((len(test_time_array), 7))
                # Initialize the filter with the first predicted joint positions
                filtered_predicted_joint_positions_over_time[0, :] = predicted_joint_positions_over_time[0, :]
                # Set the smoothing factor
                alpha = 0.1
                # Apply the filter
                for i in range(1, len(test_time_array)):
                    filtered_predicted_joint_positions_over_time[i, :] = alpha * predicted_joint_positions_over_time[i, :] + \
                                                                        (1 - alpha) * filtered_predicted_joint_positions_over_time[i - 1, :]
                # Update the predicted joint positions
                predicted_joint_positions_over_time = filtered_predicted_joint_positions_over_time

            # Compute qd_des_over_time by numerically differentiating the predicted joint positions
            qd_des_over_time = np.gradient(predicted_joint_positions_over_time, axis=0, edge_order=2) / time_step
            # Clip the joint velocities to the joint limits
            qd_des_over_time_clipped = np.clip(qd_des_over_time, -np.array(joint_vel_limits), np.array(joint_vel_limits))

            # Store the predicted joint positions and velocities
            q_pred_all_for_all_models.append(predicted_joint_positions_over_time)
            qd_pred_all_for_all_models.append(qd_des_over_time_clipped)

            # Store data
            q_mes_all = []
            q_des_all = []
            time_mes_all = []
            tau_cmd_all = []
            # Data collection loop
            while current_time < test_time_array.max():
                # Measure current state
                q_mes = sim.GetMotorAngles(0)
                qd_mes = sim.GetMotorVelocities(0)
                qdd_est = sim.ComputeMotorAccelerationTMinusOne(0)

                # Get the index corresponding to the current time
                current_index = int(current_time / time_step)
                if current_index >= len(test_time_array):
                    current_index = len(test_time_array) - 1

                # Get q_des and qd_des_clip from predicted data
                q_des = predicted_joint_positions_over_time[current_index, :]  # Shape: (7,)
                qd_des_clip = qd_des_over_time_clipped[current_index, :]      # Shape: (7,)

                # Control command
                tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des_clip, kp, kd)
                cmd.SetControlCmd(tau_cmd, ["torque"] * 7)  # Set the torque command
                sim.Step(cmd, "torque")  # Simulation step with torque command

                # Keyboard event handling
                keys = sim.GetPyBulletClient().getKeyboardEvents()
                qKey = ord('q')

                # Exit logic with 'q' key
                if qKey in keys and keys[qKey] & sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
                    print("Exiting simulation.")
                    break

                # Time management
                time.sleep(time_step)  # Control loop timing
                current_time += time_step

                time_mes_all.append(current_time)
                q_mes_all.append(q_mes)
                q_des_all.append(q_des)
                tau_cmd_all.append(tau_cmd)

            # After the trajectory, compute the final cartesian position
            final_predicted_joint_positions = predicted_joint_positions_over_time[-1, :]  # Shape: (7,)
            final_cartesian_pos, final_R = dyn_model.ComputeFK(final_predicted_joint_positions, controlled_frame_name)
            #print(f"Final computed cartesian position: {final_cartesian_pos}")
            # Compute position error
            position_error = np.linalg.norm(final_cartesian_pos - goal_position)
            print(f"Position error between computed position and goal for {test_model}: {position_error}")

            q_mes_all_for_all_models.append(q_mes_all)
            q_des_all_for_all_models.append(q_des_all)
            tau_cmd_all_for_all_models.append(tau_cmd_all)
            #print(f"Finished testing model: {neural_network_or_random_forest}-----------")

        # # Plot the measured trajectory for all models
        # q_mes_all_for_all_models = np.array(q_mes_all_for_all_models)
        # for i in range(num_joints):
        #     plt.figure(figsize=(12, 6))
        #     # Plot the measured idx i joint position for all models
        #     for j, test_model in enumerate(test_models):
        #         plt.plot(time_mes_all, q_mes_all_for_all_models[j][:, i], label=f'Measured Joint {i+1} - {test_model}')
        #     plt.title(f'Measured Joint{i+1} Positions')
        #     plt.xlabel('Time [s]')
        #     plt.ylabel('Joint Position')
        #     plt.legend()
        #     plt.show()

        # Plot the actual trajectory vs the predicted trajectory for all models
        q_mes_all_for_all_models = np.array(q_mes_all_for_all_models)
        for i in range(num_joints):
            # Plot the actual and predicted idx i joint position for all models
            for j, test_model in enumerate(test_models):
                plt.figure(figsize=(12, 6))
                plt.plot(test_time_array, q_pred_all_for_all_models[j][:, i], label=f'Predicted Joint {i+1} - {test_model}')
                plt.plot(time_mes_all, q_mes_all_for_all_models[j][:, i], label=f'Measured Joint {i+1} - {test_model}')
                plt.title(f'Predicted vs Measured Joint{i+1} Positions')
                plt.xlabel('Time [s]')
                plt.ylabel('Joint Position')
                plt.legend()
                plt.show()

        # Plot the tracking error for all models
        for i in range(num_joints):
            plt.figure(figsize=(12, 6))
            # Plot the tracking error for idx i joint for all models
            for j, test_model in enumerate(test_models):
                q_mes_all = np.array(q_mes_all_for_all_models[j])
                q_des_all = np.array(q_des_all_for_all_models[j])
                tracking_error = q_mes_all - q_des_all
                plt.plot(time_mes_all, tracking_error[:, i], label=f'Tracking Error Joint {i+1} - {test_model}')
            plt.title(f'Tracking Error Joint{i+1} Positions')
            plt.xlabel('Time [s]')
            plt.ylabel('Tracking Error')
            plt.legend()
            plt.show()
        # Print the MSE for each model
        for j, test_model in enumerate(test_models):
            q_mes_all = np.array(q_mes_all_for_all_models[j])
            q_des_all = np.array(q_des_all_for_all_models[j])
            mse = np.mean((q_mes_all - q_des_all) ** 2)
            print(f"Mean Squared Error for {test_model}: {mse}")
        
        # Plot the commanded torques for all models
        tau_cmd_all_for_all_models = np.array(tau_cmd_all_for_all_models)
        for i in range(num_joints):
            plt.figure(figsize=(12, 6))
            # Plot the commanded torque for idx i joint for all models
            for j, test_model in enumerate(test_models):
                plt.plot(time_mes_all, tau_cmd_all_for_all_models[j][:, i], label=f'Commanded Torque Joint {i+1} - {test_model}')
            plt.title(f'Commanded Torque Joint{i+1} Positions')
            plt.xlabel('Time [s]')
            plt.ylabel('Torque')
            plt.legend()
            plt.show()

        # Plot the predicted trajectory for all models
        q_pred_all_for_all_models = np.array(q_pred_all_for_all_models)
        for i in range(num_joints):
            plt.figure(figsize=(12, 6))
            # Plot the predicted idx i joint position for all models
            for j, test_model in enumerate(test_models):
                plt.plot(test_time_array, q_pred_all_for_all_models[j][:, i], label=f'Predicted Joint {i+1} - {test_model}')
            plt.title(f'Predicted Joint{i+1} Positions')
            plt.xlabel('Time [s]')
            plt.ylabel('Joint Position')
            plt.legend()
            plt.show()

        # Plot the predicted joint velocities for all models
        qd_pred_all_for_all_models = np.array(qd_pred_all_for_all_models)
        for i in range(num_joints):
            plt.figure(figsize=(12, 6))
            # Plot the predicted idx i joint velocity for all models
            for j, test_model in enumerate(test_models):
                plt.plot(test_time_array, qd_pred_all_for_all_models[j][:, i], label=f'Predicted Joint {i+1} - {test_model}')
            plt.title(f'Predicted Joint{i+1} Velocities')
            plt.xlabel('Time [s]')
            plt.ylabel('Joint Velocity')
            plt.legend()
            plt.show()
        
if __name__ == '__main__':
    main()
