import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os

def plot_training_metrics(rewards, durations, distances, epsilons, reward_breakdown, filename):
    """
    Plot training metrics with robust error handling.
    
    Args:
        rewards: List of episode rewards
        durations: List of episode durations (steps)
        distances: List of episode distances traveled
        epsilons: List of epsilon values per episode
        reward_breakdown: Dict of reward component lists
        filename: Output filename for the plot
    """
    # Validate inputs
    if not rewards or len(rewards) == 0:
        print("Warning: No rewards data to plot")
        return
    
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        plt.figure(figsize=(20, 15))

        # 1. Rewards
        plt.subplot(3, 2, 1)
        if len(rewards) > 0:
            plt.plot(rewards, color='blue', alpha=0.3)
            # Calculate moving average
            if len(rewards) >= 50:
                moving_avg = [sum(rewards[i-50:i])/50 for i in range(50, len(rewards))]
                plt.plot(range(50, len(rewards)), moving_avg, color='red', label='50-Ep Avg')
                plt.legend()
        plt.title('Rewards per Episode')
        plt.ylabel('Reward')
        plt.grid(True)

        # 2. Durations (Steps)
        plt.subplot(3, 2, 2)
        if len(durations) > 0:
            plt.plot(durations, color='green')
        plt.title('Duration (Steps) per Episode')
        plt.ylabel('Steps')
        plt.grid(True)

        # 3. Distances
        plt.subplot(3, 2, 3)
        if len(distances) > 0:
            plt.plot(distances, color='orange')
        plt.title('Distance Traveled (m)')
        plt.ylabel('Meters')
        plt.grid(True)

        # 4. Success Rate (Approximate based on Reward)
        plt.subplot(3, 2, 4)
        if len(rewards) > 0 and len(durations) > 0:
            # Binary success (1 if reward > 100, indicating goal reached)
            successes = [1 if r > 100 else 0 for r in rewards]
            if len(successes) >= 50:
                success_rate = [sum(successes[i-50:i])/50.0 for i in range(50, len(successes))]
                plt.plot(range(50, len(successes)), success_rate, color='purple')
            elif len(successes) > 0:
                plt.plot(successes, color='purple', alpha=0.5)
        plt.title('Success Rate (Rolling 50-Ep)')
        plt.ylabel('Rate (0-1)')
        plt.ylim(0, 1.1)
        plt.grid(True)
        
        # 5. Epsilon Decay
        plt.subplot(3, 2, 5)
        if len(epsilons) > 0:
            plt.plot(epsilons, color='red')
        plt.title('Epsilon Decay')
        plt.ylabel('Epsilon')
        plt.ylim(0, 1.0)
        plt.grid(True)
        
        # 6. Reward Breakdown
        plt.subplot(3, 2, 6)
        if reward_breakdown and isinstance(reward_breakdown, dict):
            for key, values in reward_breakdown.items():
                if values and len(values) > 0:
                    # Plot moving average for clarity
                    if len(values) >= 50:
                        moving_avg = [sum(values[i-50:i])/50 for i in range(50, len(values))]
                        plt.plot(range(50, len(values)), moving_avg, label=key)
                    else:
                        plt.plot(values, label=key, alpha=0.5)
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.title('Reward Components (Rolling 50-Ep)')
        plt.ylabel('Reward Value')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"Plot saved successfully to: {filename}")
        
    except Exception as e:
        print(f"Error generating plot: {e}")
        # Close any open figures to prevent memory leaks
        plt.close('all')
