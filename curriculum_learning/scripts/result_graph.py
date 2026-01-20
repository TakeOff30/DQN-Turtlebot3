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
        self.count = 1

        self.plot()

        self.ros_thread = GraphSubscriber(self)
        self.ros_thread.start()

    def receive_data(self, msg):
        self.data_list.append(msg.data[0])
        self.ep.append(self.count)
        self.count += 1
        self.rewards.append(msg.data[1])

    def plot(self):
        self.qValuePlt = pyqtgraph.PlotWidget(self, title='Average max Q-value')
        self.qValuePlt.setGeometry(0, 320, 600, 300)

        self.rewardsPlt = pyqtgraph.PlotWidget(self, title='Total reward')
        self.rewardsPlt.setGeometry(0, 10, 600, 300)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(200)

        self.show()

    def update(self):
        self.rewardsPlt.showGrid(x=True, y=True)
        self.qValuePlt.showGrid(x=True, y=True)

        self.rewardsPlt.plot(self.ep, self.data_list, pen=(255, 0, 0), clear=True)
        self.qValuePlt.plot(self.ep, self.rewards, pen=(0, 255, 0), clear=True)

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