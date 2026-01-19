class GridExplorer:
    def __init__(self, x_min=-10.0, x_max=10.0, y_min=-10.0, y_max=10.0, step_size=0.5):
        self.x_min = x_min
        self.y_min = y_min
        self.step_size = step_size
        self.x_bins = int((x_max - x_min) / step_size)
        self.y_bins = int((y_max - y_min) / step_size)
        self.visited_cells = set()
        self.current_cell = None

    def get_cell(self, x, y):
        # Convert continuous coordinates to grid indices
        x_idx = int((x - self.x_min) / self.step_size)
        y_idx = int((y - self.y_min) / self.step_size)
        return (x_idx, y_idx)

    def reset(self):
        self.visited_cells.clear()
        self.current_cell = None

    def get_reward(self, x, y):
        new_cell = self.get_cell(x, y)
        reward = 0.0

        if self.current_cell is None:
            # First step of episode
            self.visited_cells.add(new_cell)
            self.current_cell = new_cell
            return 0.0

        if new_cell == self.current_cell:
            # PENALTY: Stayed in same cell (stagnation)
            reward = -2 
        elif new_cell in self.visited_cells:
            # NEUTRAL: Moved to a cell we already visited (backtracking)
            reward = 0.0
            self.current_cell = new_cell
        else:
            # REWARD: Discovered a NEW cell!
            reward = 40.0
            self.visited_cells.add(new_cell)
            self.current_cell = new_cell

        return reward
