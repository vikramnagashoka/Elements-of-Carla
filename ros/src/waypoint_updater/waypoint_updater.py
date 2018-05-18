#!/usr/bin/env python
import math
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import cKDTree

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

'''

LOOKAHEAD_WPS = 20 # Number of waypoints we will publish. You can change this number
ACC_FACTOR = 1
MIN_DECEL_FACTOR = 0.2
MAX_VEL_FACTOR = 0.95
TL_MIN_DISTANCE = 1
MIN_BREAK_DISTANCE = 3

class WaypointUpdater(object):

    def __init__(self):
        rospy.init_node('waypoint_updater')

        # Parameters
        self.accel_limit = rospy.get_param('~/twist_controller/accel_limit', 1.0) * ACC_FACTOR
        self.min_decel = rospy.get_param('~/twist_controller/decel_limit', -5) * MIN_DECEL_FACTOR
        self.speed_limit = rospy.get_param('/waypoint_loader/velocity') * MAX_VEL_FACTOR
        self.speed_limit = self.speed_limit * 1000.0 / 3600.  # m/s

        # Subscribers
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_twist_cb)

        # Publishers
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # Node state given by subscribers
        self.pose = None
        self.original_waypoints = None
        self.waypoints = []
        self.waypoints_2d = []
        self.waypoint_tree = None
        self.euc_distances = []
        self.n_waypoints = -1
        self.traffic_waypoint = -1
        self.current_twist = None

        # Node state compute on each iteration
        self.current_velocity2 = None
        self.closest_waypoint = None
        self.dist_to_closest_waypoint = None

        self.loop()

    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            # Compute common state data for all actions.
            self.compute_state()
            # Compute action
            action, context = self.desired_action()
            if action:
                # Compute new waypoints
                lane = self.build_final_waypoints(action, context)
                self.final_waypoints_pub.publish(lane)
            rate.sleep()

    def compute_state(self):
        if self.current_twist:
            # Car velocity
            linear_vel = self.current_twist.linear
            self.current_velocity2 = linear_vel.x ** 2 +  linear_vel.y ** 2
        if self.pose and self.waypoint_tree:
            # Closest waypoint
            self.closest_waypoint = self.get_closest_waypoint_idx()
            # Distance to closest waypoint
            self.dist_to_closest_waypoint = self.euclidean_distance(
                self.pose.position,
                self.waypoints[self.closest_waypoint].pose.pose.position
            )

    def desired_action(self):
        if not (self.pose and self.waypoint_tree and self.current_twist):
            return None, {}
        if self.closest_waypoint == self.n_waypoints - 1:
            return None, {}

        # Car max break distance
        max_break_distance = max(
            -1. * self.current_velocity2 / (2 * self.min_decel),
            MIN_BREAK_DISTANCE
        )

        if self.traffic_waypoint > -1 and self.traffic_waypoint >= self.closest_waypoint:
            dist_to_tl = self.distance(self.traffic_waypoint)
            if dist_to_tl < TL_MIN_DISTANCE:
                return 'CONSTANT_VELOCITY', {
                    'velocity': 0.
                }
            if dist_to_tl - TL_MIN_DISTANCE <= max_break_distance:
                return 'SLOWDOWN', {
                    'waypoint': self.traffic_waypoint,
                    'offset': TL_MIN_DISTANCE
                }

        if self.closest_waypoint + 200 > self.n_waypoints:
            dist_to_end = self.distance(self.n_waypoints - 1)
            if dist_to_end <= max_break_distance:
                return 'SLOWDOWN', {
                    'waypoint': self.n_waypoints - 1,
                    'offset': 0
                }
        return 'ACCELERATE', {}

    def build_final_waypoints(self, action, context):
        if action == 'CONSTANT_VELOCITY':
            return self.constante_velocity_waypoints(**context)
        if action == 'SLOWDOWN':
            return self.slowdown_waypoints(**context)
        return self.accelerate_waypoints(**context)

    def constante_velocity_waypoints(self, velocity):
        end = min(self.closest_waypoint + LOOKAHEAD_WPS, self.n_waypoints)
        waypoints = [None] * (end - self.closest_waypoint)
        for idx in range(self.closest_waypoint, end):
            waypoint = Waypoint()
            waypoint.pose = self.waypoints[idx].pose
            waypoint.twist.twist.linear.x = velocity
            waypoints[idx-self.closest_waypoint] = waypoint
        return self.build_lane(waypoints)

    def slowdown_waypoints(self, waypoint, offset):
        end = min(self.closest_waypoint + LOOKAHEAD_WPS, self.n_waypoints)

        dist_to_waypoints = self.distance_list(max(waypoint, end - 1))
        distance_to_stop = dist_to_waypoints[waypoint - self.closest_waypoint] - offset

        if distance_to_stop <= 0:
            return self.constante_velocity_waypoints(0.)

        decel = self.current_velocity2/(2 * distance_to_stop)

        waypoints = [None] * (end - self.closest_waypoint)
        for idx in range(self.closest_waypoint, end):
            dist = dist_to_waypoints[idx - self.closest_waypoint]
            velocity = self.current_velocity2 - 2 * decel * dist
            if velocity < 0.1:
                velocity = 0.0
            velocity = math.sqrt(velocity)
            waypoint = Waypoint()
            waypoint.pose = self.waypoints[idx].pose
            waypoint.twist.twist.linear.x = velocity
            waypoints[idx-self.closest_waypoint] = waypoint
        return self.build_lane(waypoints)

    def accelerate_waypoints(self):
        end = min(self.closest_waypoint + LOOKAHEAD_WPS, self.n_waypoints)

        dist_to_waypoints = self.distance_list(end - 1)

        waypoints = [None] * (end - self.closest_waypoint)
        for idx in range(self.closest_waypoint, end):
            dist = dist_to_waypoints[idx - self.closest_waypoint]
            velocity = math.sqrt(self.current_velocity2 + 2 * self.accel_limit * dist)
            if velocity > self.speed_limit:
                velocity = self.speed_limit
            waypoint = Waypoint()
            waypoint.pose = self.waypoints[idx].pose
            waypoint.twist.twist.linear.x = velocity
            waypoints[idx-self.closest_waypoint] = waypoint
        return self.build_lane(waypoints)

    def build_lane(self, waypoints):
        lane = Lane()
        lane.header = self.original_waypoints.header
        lane.waypoints = waypoints
        return lane

    def pose_cb(self, msg):
        self.pose = msg.pose

    def waypoints_cb(self, lane):
        self.original_waypoints = lane
        self.waypoints = lane.waypoints
        self.n_waypoints = len(self.waypoints)

        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y]
                                 for waypoint in self.waypoints]
            self.waypoint_tree = cKDTree(self.waypoints_2d, leafsize=1)
            self.euc_distances = np.empty(self.n_waypoints-1, dtype=float)
            for i in range(self.n_waypoints-1):
                self.euc_distances[i] = self.euclidean_distance(
                    self.waypoints[i].pose.pose.position,
                    self.waypoints[i + 1].pose.pose.position
                )

    def traffic_cb(self, msg):
        self.traffic_waypoint = msg.data

    def current_twist_cb(self, msg):
        self.current_twist = msg.twist

    def get_closest_waypoint_idx(self):
        x = self.pose.position.x
        y = self.pose.position.y
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]

        # Check if closest is ahead or behind vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx-1]

        # Equation for hyperplane through closest_coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)

        if val > 0:
            closest_idx = (closest_idx + 1) % self.n_waypoints
        return closest_idx

    def euclidean_distance(self, point_a, point_b):
        return math.sqrt(
            (point_a.x-point_b.x)**2 + (point_a.y-point_b.y)**2 + (point_a.z-point_b.z)**2
        )

    def distance_list(self, wp2):
        distances = np.empty(wp2-self.closest_waypoint+1, dtype=float)
        distances[0] = self.dist_to_closest_waypoint
        distances[1:] = self.euc_distances[self.closest_waypoint:wp2]
        return distances.cumsum()

    def distance(self, wp2):
        dist = self.dist_to_closest_waypoint
        return dist + sum(self.euc_distances[self.closest_waypoint:wp2])

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
