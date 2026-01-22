from curriculum_learning import model_dir, reports_dir

class TrainingManager:
    def __init__(self, checkpoint_manager, reporter):
        self.max_episode_duration = 0
        self.max_distance_traveled = 0.0
        self.episode_rewards_history = []
        self.episode_durations_history = []
        self.episode_distances_history = []
        self.episode_epsilon_history = []
        self.reward_breakdown_history = []
        self.episode_breakdown = {
            'navigation': 0.0,
            'collision': 0.0,
            'success': 0.0,
            'failure': 0.0,
            'other': 0.0
        }
        self.checkpoint_manager = checkpoint_manager
        self.reporter = reporter
        
    def accumulate_breakdown(self, reward):
        if reward > 100:
            self.episode_breakdown['success'] += reward
        elif reward < -50:
            self.episode_breakdown['failure'] += reward
        elif reward < 0:
            self.episode_breakdown['collision'] += reward
        else:
            self.episode_breakdown['navigation'] += reward
        
    def update_metrics(self, cumulated_reward, episode_distance, t, current_eps):
        # Track metrics
        self.episode_rewards_history.append(cumulated_reward)
        self.episode_durations_history.append(t + 1)
        self.episode_distances_history.append(episode_distance)
        self.reward_breakdown_history.append(self.episode_breakdown)
        self.episode_epsilon_history.append(current_eps)
    
    def save_checkpoint_plots(self, i_episode, outdir):
        breakdown_dict = {}
        if self.reward_breakdown_history:
            # Initialize with keys from first episode
            for key in self.reward_breakdown_history[0].keys():
                breakdown_dict[key] = [ep[key] for ep in self.reward_breakdown_history]
        
        # Generate and save plot
        plot_filename = self.reporter.generate_plot(i_episode + 1, outdir, self.episode_rewards_history,
                                                self.episode_durations_history, self.episode_distances_history,
                                                self.episode_epsilon_history, breakdown_dict)
        
        return plot_filename
        