import numpy as np
import matplotlib.pyplot as plt

# =========================== CORE SYSTEM CLASSES =========================== #

# Probability Matrix (Analog Component)
class ProbabilityMatrix:
    def __init__(self, rows, cols, initial_values=None):
        """Initialize the probability matrix."""
        if initial_values is None:
            self.matrix = np.random.rand(rows, cols)  # Start with random probabilities
        else:
            self.matrix = np.array(initial_values).reshape(rows, cols)

    def apply_noise(self, noise_type='gaussian', **params):
        """Add various types of noise to the matrix."""
        noise = np.random.normal(0, params.get('level', 0.05), self.matrix.shape)
        self.matrix += noise
        self.matrix = np.clip(self.matrix, 0, 1)
        return self

    def __str__(self):
        return str(self.matrix)


# Analog Math Calculator
class AnalogCalculator:
    def __init__(self, rows, cols):
        """Simulate analog matrix math operations."""
        self.matrix = ProbabilityMatrix(rows, cols)

    def compute(self):
        """Perform a placeholder analog matrix operation."""
        self.matrix.apply_noise(noise_type='gaussian', level=0.02)
        return self.matrix


# LLM Interface Layer
class LLMInterface:
    def __init__(self):
        """Simulate a lightweight LLM interface."""
        self.context = "Default Context"

    def decide(self, feedback_state):
        """Make decisions based on feedback state."""
        if feedback_state > 0.5:
            return "Increase Stability"
        else:
            return "Increase Dynamics"

    def update_context(self, feedback_state):
        """Update context dynamically based on feedback state."""
        if feedback_state > 0.8:
            self.context = "System Overloaded"
        elif feedback_state < 0.2:
            self.context = "System Underloaded"
        else:
            self.context = "Nominal Operation"
        return self.context


# Feedback Loop Manager
class FeedbackLoopManager:
    def __init__(self):
        """Manage feedback loops and dynamic behavior."""
        self.feedback_state = None

    def analyze(self, matrix):
        """Analyze matrix outputs and generate feedback."""
        self.feedback_state = np.mean(matrix.matrix)  # Use the average matrix value as feedback
        return self.feedback_state


# Control Layer Integrating All Components
class ControlLayer:
    def __init__(self, matrix_rows=3, matrix_cols=3):
        """High-level control layer integrating all components."""
        self.analog_calculator = AnalogCalculator(matrix_rows, matrix_cols)
        self.feedback_manager = FeedbackLoopManager()
        self.llm_interface = LLMInterface()
        self.power_supply = PowerSupplyWithRegulation(regulator=VoltageRegulator(target_voltage=5.0))

        # Store data for visualizations
        self.voltage_history = []
        self.feedback_history = []
        self.llm_decision_history = []

    def run_iteration(self, iteration):
        """Run one iteration of the control layer logic."""
        # 1. Generate power supply effects
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


# Voltage Regulator for Stabilizing Power Supply
class VoltageRegulator:
    def __init__(self, target_voltage=5.0):
        """Voltage regulator with a target voltage."""
        self.target_voltage = target_voltage

    def regulate_voltage(self, current_voltage):
        """Adjust voltage to maintain target."""
        adjustment = (self.target_voltage - current_voltage) * 0.2
        return current_voltage + adjustment


# Power Supply with Voltage Regulation
class PowerSupplyWithRegulation:
    def __init__(self, nominal_voltage=5.0, regulator=None):
        """Simulate power dynamics and regulation."""
        self.nominal_voltage = nominal_voltage
        self.current_voltage = nominal_voltage
        self.regulator = regulator

    def generate_power_effects(self, iteration):
        """Generate power supply dynamics."""
        noise = np.random.normal(0, 0.02)
        self.current_voltage = self.nominal_voltage + noise
        if self.regulator:
            self.current_voltage = self.regulator.regulate_voltage(self.current_voltage)
        return self.current_voltage


# =========================== FULL SYSTEM SIMULATION =========================== #
def full_system_simulation():
    """Run a complete multi-iteration simulation of the full system."""
    control_layer = ControlLayer(matrix_rows=5, matrix_cols=5)

    # Run the system for 5 iterations and collect visualization data
    matrix_list = []
    for iteration in range(5):
        print(f"\n=== Iteration {iteration + 1} ===")
        matrix = control_layer.run_iteration(iteration)
        matrix_list.append(matrix.matrix)

    # Get visualization data
    voltage_history, feedback_history, llm_decision_history = control_layer.get_visualization_data()

    # Visualization 1: Matrix Evolution
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i, matrix in enumerate(matrix_list):
        ax = axes[i]
        ax.imshow(matrix, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f'Iteration {i + 1}')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle('Matrix Evolution Over Iterations')
    plt.show()

    # Visualization 2: Power Supply Voltage Over Iterations
    plt.figure(figsize=(10, 4))
    plt.plot(voltage_history, marker='o', label='Voltage')
    plt.axhline(5.0, color='r', linestyle='--', label='Target Voltage')
    plt.title('Power Supply Voltage Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.show()

    # Visualization 3: Feedback State Over Iterations
    plt.figure(figsize=(10, 4))
    plt.plot(feedback_history, marker='o', color='purple')
    plt.title('Feedback State Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Feedback State')
    plt.show()

# Run the enhanced system simulation
full_system_simulation()

import matplotlib.pyplot as plt

# Function to visualize power fluctuations and analyze key characteristics
def analyze_power_fluctuations(voltage_history, feedback_history, llm_decision_history):
    """Analyze and visualize power fluctuations over iterations."""

    # 1. Plot the Power Supply Voltage History
    plt.figure(figsize=(12, 6))
    plt.plot(voltage_history, marker='o', label='Power Voltage (V)')
    plt.axhline(5.0, color='r', linestyle='--', label='Target Voltage (5V)')

    # 2. Mark Significant Deviations from Target Voltage
    deviation_indices = [i for i, v in enumerate(voltage_history) if abs(v - 5.0) > 0.05]
    plt.scatter(deviation_indices, [voltage_history[i] for i in deviation_indices], color='orange', zorder=5, label='Voltage Deviations')

    # 3. Annotate LLM Decisions and Feedback States
    for i, (voltage, feedback, decision) in enumerate(zip(voltage_history, feedback_history, llm_decision_history)):
        plt.annotate(f'{decision}\nFeedback: {feedback:.2f}', 
                     (i, voltage), 
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center',
                     fontsize=9, 
                     bbox=dict(facecolor='lightblue', alpha=0.5, edgecolor='black'))

    # 4. Customize and Display Plot
    plt.title('Power Supply Voltage and LLM Decision Analysis')
    plt.xlabel('Iteration')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.grid(True)
    plt.show()


# Run Analysis on Previously Captured Data
def run_power_analysis():
    # Simulate the system again to collect voltage history
    control_layer = ControlLayer(matrix_rows=5, matrix_cols=5)
    matrix_list = []

    # Run the system for 5 iterations and collect data for analysis
    for iteration in range(5):
        matrix = control_layer.run_iteration(iteration)
        matrix_list.append(matrix)

    # Get visualization data
    voltage_history, feedback_history, llm_decision_history = control_layer.get_visualization_data()

    # Perform power fluctuation analysis and visualization
    analyze_power_fluctuations(voltage_history, feedback_history, llm_decision_history)


# Run the power fluctuation analysis
run_power_analysis()
# Re-run the Extended Simulation for More Iterations and Improved Visualization Handling

# Re-import necessary libraries in case of environment reset
import matplotlib.pyplot as plt
import numpy as np

# Function to visualize power fluctuations and analyze key characteristics
def analyze_power_fluctuations(voltage_history, feedback_history, llm_decision_history):
    """Analyze and visualize power fluctuations over iterations."""
    plt.figure(figsize=(14, 6))
    plt.plot(voltage_history, marker='o', label='Power Voltage (V)')
    plt.axhline(5.0, color='r', linestyle='--', label='Target Voltage (5V)')

    # Mark Significant Deviations from Target Voltage
    deviation_indices = [i for i, v in enumerate(voltage_history) if abs(v - 5.0) > 0.05]
    plt.scatter(deviation_indices, [voltage_history[i] for i in deviation_indices], color='orange', zorder=5, label='Voltage Deviations')

    # Annotate LLM Decisions and Feedback States
    for i, (voltage, feedback, decision) in enumerate(zip(voltage_history, feedback_history, llm_decision_history)):
        plt.annotate(f'{decision}\nFeedback: {feedback:.2f}', 
                     (i, voltage), 
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center',
                     fontsize=9, 
                     bbox=dict(facecolor='lightblue', alpha=0.5, edgecolor='black'))

    # Customize and Display Plot
    plt.title('Power Supply Voltage and LLM Decision Analysis Over Extended Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.grid(True)
    plt.show()


# Extended Full System Simulation with More Iterations
def extended_system_simulation(iterations=15):
    """Run a complete extended simulation of the full system for more iterations."""
    control_layer = ControlLayer(matrix_rows=5, matrix_cols=5)

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


# Run the extended simulation again for 15 iterations
extended_system_simulation(iterations=50)
