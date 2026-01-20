#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################
#
# Authors: Ryan Shim, Gilbert, ChanHyeong Lee

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
from rclpy.node import Node
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

        # Reset all action bars
        self.qt_thread.signal_action0.emit(0)
        self.qt_thread.signal_action1.emit(0)
        self.qt_thread.signal_action2.emit(0)

        # Highlight the action taken (0=FORWARD, 1=LEFT, 2=RIGHT)
        if data[0] == 0:
            self.qt_thread.signal_action0.emit(100)
        elif data[0] == 1:
            self.qt_thread.signal_action1.emit(100)
        elif data[0] == 2:
            self.qt_thread.signal_action2.emit(100)

        # Update reward displays
        if len(data) >= 3:
            self.qt_thread.signal_total_reward.emit(str(round(data[1], 2)))
            self.qt_thread.signal_reward.emit(str(round(data[2], 2)))


class Thread(QThread):

    signal_action0 = pyqtSignal(int)
    signal_action1 = pyqtSignal(int)
    signal_action2 = pyqtSignal(int)
    signal_total_reward = pyqtSignal(str)
    signal_reward = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.subscriber = None

    def run(self):
        self.subscriber = Ros1Subscriber(self)
        rospy.spin()

class Form(QWidget):

    def __init__(self, qt_thread):
        super().__init__(flags=Qt.Widget)
        self.qt_thread = qt_thread
        self.setWindowTitle('Action State - TurtleBot3 DQN')

        layout = QGridLayout()

        self.pgsb1 = QProgressBar()
        self.pgsb1.setOrientation(Qt.Vertical)
        self.pgsb1.setValue(0)
        self.pgsb1.setRange(0, 100)

        self.pgsb2 = QProgressBar()
        self.pgsb2.setOrientation(Qt.Vertical)
        self.pgsb2.setValue(0)
        self.pgsb2.setRange(0, 100)

        self.pgsb3 = QProgressBar()
        self.pgsb3.setOrientation(Qt.Vertical)
        self.pgsb3.setValue(0)
        self.pgsb3.setRange(0, 100)

        self.label_total_reward = QLabel('Total reward')
        self.edit_total_reward = QLineEdit('')
        self.edit_total_reward.setDisabled(True)
        self.edit_total_reward.setFixedWidth(100)

        self.label_reward = QLabel('Step reward')
        self.edit_reward = QLineEdit('')
        self.edit_reward.setDisabled(True)
        self.edit_reward.setFixedWidth(100)

        self.label_forward = QLabel('Forward')
        self.label_left = QLabel('Turn Left')
        self.label_right = QLabel('Turn Right')

        layout.addWidget(self.label_total_reward, 0, 0)
        layout.addWidget(self.edit_total_reward, 1, 0)
        layout.addWidget(self.label_reward, 2, 0)
        layout.addWidget(self.edit_reward, 3, 0)

        layout.addWidget(self.pgsb1, 0, 4, 4, 1)
        layout.addWidget(self.pgsb2, 0, 5, 4, 1)
        layout.addWidget(self.pgsb3, 0, 6, 4, 1)

        layout.addWidget(self.label_left, 4, 4)
        layout.addWidget(self.label_forward, 4, 5)
        layout.addWidget(self.label_right, 4, 6)

        self.setLayout(layout)

        qt_thread.signal_action0.connect(self.pgsb1.setValue)
        qt_thread.signal_action1.connect(self.pgsb2.setValue)
        qt_thread.signal_action2.connect(self.pgsb3.setValue)
        qt_thread.signal_total_reward.connect(self.edit_total_reward.setText)
        qt_thread.signal_reward.connect(self.edit_reward.setText)

    def closeEvent(self, event):
        rospy.signal_shutdown("Window closed")
        event.accept()


def run_qt_app(qt_thread):
    app = QApplication(sys.argv)
    form = Form(qt_thread)
    form.show()
    app.exec_()


def main():
    rospy.init_node("action_graph_node")
    qt_thread = Thread()
    qt_thread.start()
    
    app =QApplication(sys.argv)
    form = Form(qt_thread)
    form.show()

    def shutdown_handler(sig, frame):
        print('shutdown')
        rospy.signal_shutdown('SIGINT received')
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()