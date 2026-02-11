#!/usr/bin/env python3
import signal
import sys
import threading
import time

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtWidgets import QProgressBar
from PyQt5.QtWidgets import QWidget
import rospy
import pyqtgraph
from std_msgs.msg import Float32MultiArray


class Ros1Subscriber:

    def __init__(self, qt_thread):
        self.qt_thread = qt_thread
        self.sub = rospy.Subscriber(
            '/get_action',
            Float32MultiArray,
            self.get_array_callback
        )
        
    def get_array_callback(self, msg):
        data = list(msg.data)

        # Emit the action index to update the bar plot
        action_idx = int(data[0])
        self.qt_thread.signal_action_selected.emit(action_idx)

        if len(data) >= 2:
            self.qt_thread.signal_total_reward.emit(str(round(data[-2], 2)))
            self.qt_thread.signal_reward.emit(str(round(data[-1], 2)))


class Thread(QThread):

    signal_action_selected = pyqtSignal(int)
    signal_total_reward = pyqtSignal(str)
    signal_reward = pyqtSignal(str)

    def __init__(self):
        super().__init__()

    def run(self):
        self.subscriber = Ros1Subscriber(self)
        rospy.spin()


class Form(QWidget):

    def __init__(self, qt_thread):
        super().__init__(flags=Qt.Widget)
        self.qt_thread = qt_thread
        self.setWindowTitle('Action State')

        layout = QGridLayout()

        # Initialize action counters
        self.action_counts = [0, 0, 0, 0, 0]  # 5 actions (adjust if different)
        self.total_actions = 0

        # Create bar plot for action distribution
        self.action_plot = pyqtgraph.PlotWidget(title='Action Distribution (%)')
        self.action_plot.setMinimumHeight(300)
        self.action_plot.setYRange(0, 100)
        self.action_plot.showGrid(x=True, y=True)
        self.action_plot.setLabel('left', 'Percentage (%)')
        self.action_plot.setLabel('bottom', 'Action')
        
        # Configure x-axis with action labels including angular velocities
        action_labels = [
            'Action 0\n(+1.5 rad/s)', 
            'Action 1\n(+0.75 rad/s)', 
            'Action 2\n(0.0 rad/s)', 
            'Action 3\n(-0.75 rad/s)', 
            'Action 4\n(-1.5 rad/s)'
        ]
        x_dict = dict(enumerate(action_labels))
        ax = self.action_plot.getAxis('bottom')
        ax.setTicks([list(x_dict.items())])
        
        self.bar_graph = None

        self.label_total_reward = QLabel('Total reward')
        self.edit_total_reward = QLineEdit('')
        self.edit_total_reward.setDisabled(True)
        self.edit_total_reward.setFixedWidth(100)

        self.label_reward = QLabel('Reward')
        self.edit_reward = QLineEdit('')
        self.edit_reward.setDisabled(True)
        self.edit_reward.setFixedWidth(100)

        layout.addWidget(self.label_total_reward, 0, 0)
        layout.addWidget(self.edit_total_reward, 1, 0)
        layout.addWidget(self.label_reward, 2, 0)
        layout.addWidget(self.edit_reward, 3, 0)
        layout.addWidget(self.action_plot, 0, 1, 4, 1)

        self.setLayout(layout)

        qt_thread.signal_action_selected.connect(self.update_action_distribution)
        qt_thread.signal_total_reward.connect(self.edit_total_reward.setText)
        qt_thread.signal_reward.connect(self.edit_reward.setText)

    def update_action_distribution(self, action_idx):
        """Update action counts and refresh the bar plot"""
        if 0 <= action_idx < len(self.action_counts):
            self.action_counts[action_idx] += 1
            self.total_actions += 1
            
            # Calculate percentages
            percentages = [(count / self.total_actions * 100) if self.total_actions > 0 else 0 
                          for count in self.action_counts]
            
            # Update bar plot
            x = list(range(len(self.action_counts)))
            self.action_plot.clear()
            self.bar_graph = pyqtgraph.BarGraphItem(
                x=x, 
                height=percentages, 
                width=0.6, 
                brush='b'
            )
            self.action_plot.addItem(self.bar_graph)

    def closeEvent(self, event):
        rospy.signal_shutdown("SIGINT")
        event.accept()


def run_qt_app(qt_thread):
    app = QApplication(sys.argv)
    form = Form(qt_thread)
    form.show()
    app.exec_()


def main():
    rospy.init_node("action_graph_node", anonymous=True)

    qt_thread = Thread()
    qt_thread.start()

    app = QApplication(sys.argv)
    form = Form(qt_thread)
    form.show()

    def shutdown_handler(sig, frame):
        rospy.signal_shutdown("SIGINT")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    sys.exit(app.exec_())
    
if __name__ == '__main__':
    main()