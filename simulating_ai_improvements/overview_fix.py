import numpy as np
import matplotlib.pyplot as plt

# =========================== PID Controller Implementation =========================== #
class PIDController:
    def __init__(self, Kp=1.0, Ki=0.1, Kd=0.01, setpoint=5.0):
        """
        PID Controller Initialization.
        Kp: Proportional gain
        Ki: Integral gain
        Kd: Derivative gain
        setpoint: Desired voltage value to maintain
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint  # Target voltage level (e.g., 5.0V)

        # Internal state for PID calculations
        self.previous_error = 0
        self.integral = 0

    def compute(self, current_value):
        """
        Calculate the PID control signal based on the current value.
        current_value: The measured voltage level
        Returns: Control signal to adjust the voltage
        """
        # Calculate the error (difference between desired setpoint and current value)
        error = self.setpoint - current_value

        # Calculate integral and derivative components
        self.integral += error  # Accumulate the integral
        derivative = error - self.previous_error

        # Compute the PID output
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        # Store the current error for the next iteration
        self.previous_error = error

        return output


# =========================== System Component Adjustments =========================== #

# Adjusted Voltage Regulator Class with PID Controller
class VoltageRegulator:
    def __init__(self, target_voltage=5.0, Kp=1.0, Ki=0.1, Kd=0.01):
        """
        Voltage Regulator using PID Controller.
        target_voltage: The desired voltage level to maintain.
        Kp, Ki, Kd: PID coefficients for tuning.
        """
        self.target_voltage = target_voltage
        self.pid_controller = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, setpoint=target_voltage)

    def regulate_voltage(self, current_voltage):
        """Use the PID controller to adjust the voltage."""
        # Calculate the adjustment using the PID controller
        adjustment = self.pid_controller.compute(current_voltage)
        return current_voltage + adjustment


# =========================== System Control Layer with PID Integration =========================== #
class ControlLayer:
    def __init__(self, matrix_rows=3, matrix_cols=3, Kp=1.0, Ki=0.1, Kd=0.01):
        """High-level control layer integrating all components with PID voltage regulation."""
        self.analog_calculator = AnalogCalculator(matrix_rows, matrix_cols)
        self.feedback_manager = FeedbackLoopManager()
        self.llm_interface = LLMInterface()
        self.power_supply = PowerSupplyWithRegulation(regulator=VoltageRegulator(Kp=Kp, Ki=Ki, Kd=Kd))

        # Store data for visualizations
        self.voltage_history = []
        self.feedback_history = []
        self.llm_decision_history = []

    def run_iteration(self, iteration):
        """Run one iteration of the control layer logic."""
        # 1. Generate power supply effects with PID-controlled voltage regulation
        power_voltage = self.power_supply.generate_power_effects(iteration)
        self.voltage_history.append(power_voltage)

        # 2. Perform matrix computations using the analog calculator
        matrix = self.analog_calculator.compute()
        
        # 3. Generate feedback based on matrix outputs
        feedback_state = self.feedback_manager.analyze(matrix)
        self.feedback_history.append(feedback_state)

        # 4. Update LLM context and get decision
        context = self.llm_interface.update_context(feedback_state)
        decision = self.llm_interface.decide(feedback_state)
        self.llm_decision_history.append(decision)

        # Return matrix for visualization
        return matrix

    def get_visualization_data(self):
        """Return the data for visualization."""
        return self.voltage_history, self.feedback_history, self.llm_decision_history


# =========================== Run the Extended Simulation with PID Integration =========================== #
def extended_system_simulation_with_pid(iterations=15):
    """Run a complete extended simulation of the full system with a PID-controlled voltage regulator."""
    # Using the adjusted ControlLayer with PID settings
    control_layer = ControlLayer(matrix_rows=5, matrix_cols=5, Kp=0.8, Ki=0.2, Kd=0.05)  # Adjusted PID coefficients

    # Run the system for the specified number of iterations and collect data
    matrix_list = []
    for iteration in range(iterations):
        print(f"\n=== Iteration {iteration + 1} ===")
        matrix = control_layer.run_iteration(iteration)
        matrix_list.append(matrix.matrix)

    # Get visualization data
    voltage_history, feedback_history, llm_decision_history = control_layer.get_visualization_data()

    # Visualization 1: Matrix Evolution
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    step = max(1, iterations // 5)
    for i, matrix in enumerate(matrix_list[::step]):
        ax = axes[i]
        ax.imshow(matrix, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f'Iteration {i * step + 1}')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle(f'Matrix Evolution Over {iterations} Iterations (Sampled Every {step} Iterations)')
    plt.show()

    # Visualization 2: Power Supply Voltage Over Extended Iterations
    analyze_power_fluctuations(voltage_history, feedback_history, llm_decision_history)


# =========================== Execute the Simulation with PID =========================== #
extended_system_simulation_with_pid(iterations=50)
