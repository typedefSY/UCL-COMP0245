from skopt import gp_minimize
from skopt.space import Real
import numpy as np
import os
from matplotlib import pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference
from skopt import gp_minimize
from skopt.space import Real
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import RBF


# Configuration for the simulation
conf_file_name = "pandaconfig.json"  # Configuration file for the robot
cur_dir = os.path.dirname(os.path.abspath(__file__))
sim = pb.SimInterface(conf_file_name, conf_file_path_ext = cur_dir)  # Initialize simulation interface

# Get active joint names from the simulation
ext_names = sim.getNameActiveJoints()
ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

source_names = ["pybullet"]  # Define the source for dynamic modeling

# Create a dynamic model of the robot
dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False,0,cur_dir)
num_joints = dyn_model.getNumberofActuatedJoints()

init_joint_angles = sim.GetInitMotorAngles()

print(f"Initial joint angles: {init_joint_angles}")

# Sinusoidal reference
# Specify different amplitude values for each joint
amplitudes = [np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4]  # Example amplitudes for joints
# Specify different frequency values for each joint
frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]  # Example frequencies for joints

# Convert lists to NumPy arrays for easier manipulation in computations
amplitude = np.array(amplitudes)
frequency = np.array(frequencies)
ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())  # Initialize the reference


# Global lists to store data
kp0_values = []
kd0_values = []
tracking_errors = []

def simulate_with_given_pid_values(sim_, kp, kd, episode_duration=20, record_trajectory=False):
    
    # here we reset the simulator each time we start a new test
    sim_.ResetPose()
    
    # IMPORTANT: to ensure that no side effect happens, we need to copy the initial joint angles
    q_des = init_joint_angles.copy()
    qd_des = np.array([0]*dyn_model.getNumberofActuatedJoints())

    time_step = sim_.GetTimeStep()
    current_time = 0
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors

    # Initialize data storage
    q_mes_all, qd_mes_all, q_d_all, qd_d_all,  = [], [], [], []
    trajectory_data = {'time': [], 'q_des': [], 'q_mes': []}
    
    steps = int(episode_duration/time_step)
    # testing loop
    for _ in range(steps):
        # measure current state
        q_mes = sim_.GetMotorAngles(0)
        qd_mes = sim_.GetMotorVelocities(0)
    
        # Compute sinusoidal reference trajectory
        q_des, qd_des = ref.get_values(current_time)  # Desired position and velocity
        # Control command
        cmd.tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des, kp, kd)  # Zero torque command
        sim_.Step(cmd, "torque")  # Simulation step with torque command

        # Exit logic with 'q' key
        keys = sim_.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim_.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break
        
        #simulation_time = sim.GetTimeSinceReset()
        if record_trajectory:
            trajectory_data['time'].append(current_time)
            trajectory_data['q_des'].append(q_des)
            trajectory_data['q_mes'].append(q_mes)
        else:
            # Store data for plotting
            q_mes_all.append(q_mes)
            qd_mes_all.append(qd_mes)
            q_d_all.append(q_des)
            qd_d_all.append(qd_des)
        
        #time.sleep(0.01)  # Slow down the loop for better visualization
        # get real time
        current_time += time_step
       
    if record_trajectory:
        return trajectory_data
    # Calculate tracking error
    q_mes_all = np.array(q_mes_all)
    q_des_all = np.array(q_d_all)

    tracking_error = np.sum((q_mes_all - q_des_all)**2)  # Sum of squared error as the objective
    # print PD gains
    print("kp: ", kp)
    print("kd: ", kd)
    # print tracking error
    print(f"\033[91mTracking error: {tracking_error}\033[0m")
    
    return tracking_error


# Objective function for optimization
def objective(params):
    kp = np.array(params[:7])  # First 7 elements correspond to kp
    kd = np.array(params[7:])  # Last 7 elements correspond to kd
    episode_duration = 10
    
    tracking_error = simulate_with_given_pid_values(sim, kp, kd, episode_duration)
    
    # Collect data for the first kp and kd
    kp0_values.append(kp[0])
    kd0_values.append(kd[0])
    tracking_errors.append(tracking_error)

    return tracking_error


def main():
    # Define the search space for Kp and Kd
    # Define the search space as before
    space = [
        Real(0.1, 1000, name=f'kp{i}') for i in range(7)
    ] + [
        Real(0.0, 100, name=f'kd{i}') for i in range(7)
    ]

    rbf_kernel = RBF(
        length_scale=1.0,            # Initial length scale
        length_scale_bounds=(1e-2, 1e1)  # Bounds for length scale
    )

    gp = GaussianProcessRegressor(
        kernel=rbf_kernel,
        normalize_y=True,
        n_restarts_optimizer=5  # Optional for better hyperparameter optimization
        )
    
    acq_func='EI'  # Expected Improvement

    # Perform Bayesian optimization
    result = gp_minimize(
        objective,
        space,
        n_calls=100,
        base_estimator=gp,  # Use the custom Gaussian Process Regressor
        acq_func=acq_func, # TODO change this LCB': Lower Confidence Bound 'EI': Expected Improvement 'PI': Probability of Improvement
        acq_optimizer='auto',
        n_jobs=-15,
        random_state=42
    )
    
    # Extract the optimal values
    best_kp = result.x[:7]  # Optimal kp vector
    best_kd = result.x[7:]  # Optimal kd vector
    
    print(f"Optimal Kp: {best_kp}\nOptimal Kd: {best_kd}")
    
    trajectory_data = simulate_with_given_pid_values(sim, np.array(best_kp), np.array(best_kd), record_trajectory=True)
    
    # plot tracking errors
    plt.figure(figsize=(12, 6))
    plt.plot(tracking_errors, label='Tracking Error')
    window_size = 5
    mean_errors = [
        np.mean(tracking_errors[i:i + window_size]) 
        for i in range(0, len(tracking_errors), window_size)
    ]
    mean_iterations = np.arange(0, len(tracking_errors), window_size) + window_size // 2
    plt.plot(mean_iterations, mean_errors, 'r-', label="Mean Error (every 5 iterations)")
    plt.xlabel('Iteration')
    plt.ylabel('Tracking Error')
    plt.title(f'Tracking Error vs Iteration, {acq_func}')
    plt.grid()
    plt.legend()
    # !Uncomment the following lines to save the plot
    if not os.path.exists('images'):
        os.makedirs('images')
    file_path = f'images/tracking_error_vs_iteration_{acq_func}.png'
    plt.savefig(file_path)
    
    plt.figure(figsize=(15, 45))
    time_steps = trajectory_data['time']
    for i in range(num_joints):
        plt.subplot(num_joints, 1, i + 1)
        q_des_joint = np.array([qd[i] for qd in trajectory_data['q_des']])
        q_mes_joint = np.array([qm[i] for qm in trajectory_data['q_mes']])
        plt.plot(time_steps, q_des_joint, 'b-', label=f'Joint {i+1} Desired')
        plt.plot(time_steps, q_mes_joint, 'r--', label=f'Joint {i+1} Measured')
        plt.ylabel(f'Joint {i+1} Position')
        plt.legend()
    plt.xlabel('Time Steps')
    plt.suptitle(f'Desired vs Measured Joint Position optimized using {acq_func}')
    # !Uncomment the following lines to save the plot
    if not os.path.exists('images'):
        os.makedirs('images')
    file_path = f'images/desired_vs_measured_joint_position_{acq_func}.png'
    plt.savefig(file_path)
    plt.show()

    # Prepare data
    # kp0_values_array = np.array(kp0_values).reshape(-1, 1)
    # kd0_values_array = np.array(kd0_values).reshape(-1, 1)
    # tracking_errors_array = np.array(tracking_errors)

    # Fit GP models
    # gp_kp0 = fit_gp_model_1d(kp0_values_array, tracking_errors_array)
    # gp_kd0 = fit_gp_model_1d(kd0_values_array, tracking_errors_array)
    
    # Plot the results
    # plot_gp_results_1d(kp0_values_array, kd0_values_array, tracking_errors_array, gp_kp0, gp_kd0, acq_func)

if __name__ == "__main__":
    main()