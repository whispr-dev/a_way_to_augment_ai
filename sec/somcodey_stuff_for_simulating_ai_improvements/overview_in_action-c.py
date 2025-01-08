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
