import matplotlib.pyplot as plt

def plot_training_metrics(rewards, durations, distances, epsilons, reward_breakdown, filename):
    plt.figure(figsize=(20, 15))

    # 1. Rewards
    plt.subplot(3, 2, 1)
    plt.plot(rewards, color='blue', alpha=0.3)
    # Calculate moving average
    if len(rewards) >= 50:
        moving_avg = [sum(rewards[i-50:i])/50 for i in range(50, len(rewards))]
        plt.plot(range(50, len(rewards)), moving_avg, color='red', label='50-Ep Avg')
    plt.title('Rewards per Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)

    # 2. Durations (Steps)
    plt.subplot(3, 2, 2)
    plt.plot(durations, color='green')
    plt.title('Duration (Steps) per Episode')
    plt.ylabel('Steps')
    plt.grid(True)

    # 3. Distances
    plt.subplot(3, 2, 3)
    plt.plot(distances, color='orange')
    plt.title('Distance Traveled (m)')
    plt.ylabel('Meters')
    plt.grid(True)

    # 4. Success Rate (Approximate based on Reward/Duration)
    plt.subplot(3, 2, 4)
    # Binary success (1 if survived 800 steps or high reward, 0 otherwise)
    # We use a simple heuristic here, but we could use the 'survival' component from breakdown
    successes = [1 if (d >= 799 or r > 400) else 0 for d, r in zip(durations, rewards)]
    if len(successes) >= 50:
        success_rate = [sum(successes[i-50:i])/50.0 for i in range(50, len(successes))]
        plt.plot(range(50, len(successes)), success_rate, color='purple')
    plt.title('Success Rate (Rolling 50-Ep)')
    plt.ylabel('Rate (0-1)')
    plt.ylim(0, 1.1)
    plt.grid(True)
    
    # 5. Reward Breakdown
    plt.subplot(3, 1, 3)
    for key, values in reward_breakdown.items():
        if len(values) > 0:
            # Plot moving average for clarity
            if len(values) >= 50:
                moving_avg = [sum(values[i-50:i])/50 for i in range(50, len(values))]
                plt.plot(range(50, len(values)), moving_avg, label=key)
            else:
                plt.plot(values, label=key, alpha=0.5)
    
    plt.title('Reward Components Contribution (Rolling 50-Ep)')
    plt.ylabel('Reward Value')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
