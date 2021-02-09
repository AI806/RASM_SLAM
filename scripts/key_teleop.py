#! /usr/bin/env python
import rospy, math
import numpy as np
import sys, termios, tty, select, os
from geometry_msgs.msg import Twist
 
class KeyTeleop(object):
  cmd_bindings = {'q':np.array([0.3, 0.3]),
                  'w':np.array([1, 0]),
                  'e':np.array([-0.3,-0.3]),
                  'a':np.array([0, 0.4]),
                  'd':np.array([0, -0.4]),
                  'z':np.array([-0.3, -0.3]),
                  'x':np.array([-1, 0]),
                  'c':np.array([-0.3, 0.3])
                  }
  set_bindings = { 't':np.array([1,1]),
                  'b':np.array([-1,-1]),
                  'y':np.array([1,0]),
                  'n':np.array([-1,0]),
                  'u':np.array([0,1]),
                  'm':np.array([0,-1])
                }
  def init(self):
    # Save terminal settings
    self.settings = termios.tcgetattr(sys.stdin)
    # Initial values
    self.inc_ratio = 0.1
    self.speed = np.array([0.5, 1.0])
    self.command = np.array([0, 0])
    self.update_rate = 10   # Hz
    self.alive = True
    # Setup publisher
    self.pub_twist = rospy.Publisher('/cmd_vel', Twist)
    self.acc = np.array([1.0, np.pi * 30 / 180.0 ])
    self.inAccing = False
    self.curSpeed = np.array([0, 0])
    self.deltaTime = 0.1

  def fini(self):
    # Restore terminal settings
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
 
  def run(self):
    # try:
    self.init()
    print(" init the key control")
    self.print_usage()
    r = rospy.Rate(self.update_rate) # Hz
    stop = False

    while not rospy.is_shutdown() and stop == False:
      if self.inAccing == False:
        self.ch = self.get_key()
        self.inAccing = True

      self.process_key()
      self.update()
      # twist = Twist()
      # cmd  = self.speed*self.command
      # twist.linear.x = 0.1
      # twist.angular.z = 0
      # self.pub_twist.publish(twist)

      if self.ch == 'g':
        stop = True
      r.sleep()
      

    print(" end the key control")
    # except rospy.exceptions.ROSInterruptException:
    #   pass
    # finally:
    self.fini()
 
  def print_usage(self):
    msg = """
    Keyboard Teleop that Publish to /cmd_vel (geometry_msgs/Twist)
    Copyright (C) 2013
    Released under BSD License
    --------------------------------------------------
    H:       Print this menu
    Moving around:
      Q   W   E
      A   S   D
      Z   X   Z
    T/B :   increase/decrease max speeds 10%
    Y/N :   increase/decrease only linear speed 10%
    U/M :   increase/decrease only angular speed 10%
    anything else : stop
 
    G :   Quit
    --------------------------------------------------
    """
    # self.loginfo(msg)
    self.show_status()
 
  # Used to print items to screen, while terminal is in funky mode
  def loginfo(self, str):
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
    print(str)
    tty.setraw(sys.stdin.fileno())
 
  # Used to print teleop status
  def show_status(self):
    msg = 'Status:\tlinear %.2f\tangular %.2f' % (self.speed[0],self.speed[1])
    # self.loginfo(msg)
 
  # For everything that can't be a binding, use if/elif instead
  def process_key(self):
    if self. == 'h':
      self.print_usage()
    elif self. in self.cmd_bindings.keys():
      self.command = self.cmd_bindings[self.]
    elif self. in self.set_bindings.keys():
      self.speed = self.speed * (1 + self.set_bindings[self.]*self.inc_ratio)
      self.show_status()     
    elif self. == 'g':
      # self.loginfo('Quitting')
      # Stop the robot
      twist = Twist()
      self.pub_twist.publish(twist)
      # rospy.signal_shutdown('Shutdown')
    else:
      self.command = np.array([0, 0])
 
  def update(self):
    if rospy.is_shutdown():
      return
    speedReach = False
    angularReach = False

    if self.command[0] > 0:
      self.curSpeed[0]  = self.curSpeed[0] + self.deltaTime * self.acc
    
      if self.curSpeed[0] > self.command[0]:
        self.curSpeed[0] = self.command[0]
        speedReach = True
    else if self.command[0] < 0:
      self.curSpeed[0]  = self.curSpeed[0] - self.deltaTime * self.acc
    
      if self.curSpeed[0] < self.command[0]:
        self.curSpeed[0] = self.command[0]
        speedReach = True
    else:
      

    if self.curSpeed[1] > self.command[1]:
      self.curSpeed[1] = self.command[1]
      angularReach = True
    if angularReach and speedReach:
      self.inAccing = False

    twist = Twist()
    twist.linear.x = self.curSpeed[0]
    twist.angular.z = self.curSpeed[1]
    self.pub_twist.publish(twist)
 
  # Get input from the terminal
  def get_key(self):
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0)
    key = sys.stdin.read(1)
    return key.lower()
 
if __name__ == '__main__':
  rospy.init_node('keyboard_teleop')
  teleop = KeyTeleop()
  teleop.run()