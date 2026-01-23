#!/usr/bin/env python3
import signal
import sys
import threading

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
import pyqtgraph
import rospy
from std_msgs.msg import Float32MultiArray


class GraphSubscriber(threading.Thread):

    def __init__(self, window):
        super(GraphSubscriber, self).__init__()

        self.window = window
        self.daemon = True
        
    def run(self):
        
        rospy.Subscriber(
            '/result',
            Float32MultiArray,
            self.data_callback
        )
        rospy.spin()

    def data_callback(self, msg):
        self.window.receive_data(msg)


class Window(QMainWindow):

    def __init__(self):
        super(Window, self).__init__()

        self.setWindowTitle('Result')
        self.setGeometry(50, 50, 600, 650)

        self.ep = []
        self.data_list = []
        self.rewards = []
        self.avg_rewards = []
        self.count = 1
        self.losses = []
        self.loss_ep = []

        self.plot()

        self.ros_thread = GraphSubscriber(self)
        self.ros_thread.start()

    def receive_data(self, msg):
        # msg.data: [avg_max_q, total_reward, (optional) loss]
        self.data_list.append(msg.data[0])
        self.ep.append(self.count)
        self.count += 1
        self.rewards.append(msg.data[1])

        # --- CALCOLO MEDIA MOBILE (ultimi 20 episodi) ---
        window_size = 20
        if len(self.rewards) >= window_size:
            avg = sum(self.rewards[-window_size:]) / window_size
        else:
            avg = sum(self.rewards) / len(self.rewards)
        self.avg_rewards.append(avg)
        # -----------------------------------------------

        # If loss is provided, append it
        if len(msg.data) > 2:
            self.losses.append(msg.data[2])
            self.loss_ep.append(self.count-1)

    def plot(self):
        self.qValuePlt = pyqtgraph.PlotWidget(self, title='Average max Q-value')
        self.qValuePlt.setGeometry(0, 320, 600, 150)

        self.rewardsPlt = pyqtgraph.PlotWidget(self, title='Total reward')
        self.rewardsPlt.setGeometry(0, 10, 600, 150)

        self.lossPlt = pyqtgraph.PlotWidget(self, title='Loss')
        self.lossPlt.setGeometry(0, 170, 600, 140)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(200)

        self.show()

    def update(self):
        self.rewardsPlt.showGrid(x=True, y=True)
        self.qValuePlt.showGrid(x=True, y=True)
        self.lossPlt.showGrid(x=True, y=True)

        self.rewardsPlt.plot(self.ep, self.rewards, pen=(255, 0, 0),  clear=True)
        self.rewardsPlt.plot(self.ep, self.avg_rewards, pen=(0, 255, 255), width=2)

        self.qValuePlt.plot(self.ep, self.data_list, pen=(0, 255, 0), clear=True)

        if self.losses:
            self.lossPlt.plot(self.loss_ep, self.losses, pen=(255, 165, 0), clear=True)

    def closeEvent(self, event):
        rospy.signal_shutdown("Window closed")
        event.accept()


def main():
    rospy.init_node('result_graph_node',anonymous=True)
    app = QApplication(sys.argv)
    win = Window()

    def shutdown_handler(sig, frame):
        print('shutdown')
        rospy.signal_shutdown('SIGINT received')
        app.quit()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    sys.exit(app.exec())


if __name__ == '__main__':
    main()