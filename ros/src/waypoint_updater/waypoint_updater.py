#!/usr/bin/env python
import math
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree


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
ACC_FACTOR = 0.8
MAX_VEL_FACTOR = 0.95


class WaypointUpdater(object):

    def __init__(self):
        rospy.loginfo("WUP: Waypoint Updater initialization\n")
        rospy.init_node('waypoint_updater')

        self.accel_limit = rospy.get_param('~/twist_controller/accel_limit', 1.0) * ACC_FACTOR
        self.speed_limit = rospy.get_param('/waypoint_updater/speed_limit') * MAX_VEL_FACTOR
        self.speed_limit = self.speed_limit * 1609.34 / 3600.  # m/s

        # Subscribers
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)

        # Publishers
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.pose = None
        self.original_waypoints = None
        self.waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.traffic_waypoint = None
        self.current_velocity = None

        self.loop()

    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            action, context = self.desired_action()
            if action:
                lane = self.build_final_waypoints(action, context)
                self.final_waypoints_pub.publish(lane)
            rate.sleep()

    def desired_action(self):
        if not (self.pose and self.waypoint_tree and self.current_velocity):
            return None, None
        if self.traffic_waypoint is not None and self.traffic_waypoint > -1:
            return 'SLOWDOWN', self.traffic_waypoint - 2
        return 'ACCELERATE', None

    def build_final_waypoints(self, action, context):
        if action == 'SLOWDOWN':
            return self.slowdown_waypoints(context)
        return self.accelerate_waypoints(context)

    def accelerate_waypoints(self, _):
        linear_vel = self.current_velocity.linear
        init_vel = math.sqrt(linear_vel.x ** 2 +  linear_vel.y ** 2)

        closest_waypoint = self.get_closest_waypoint_idx()

        end = min(closest_waypoint + LOOKAHEAD_WPS, len(self.waypoints) - 1)
        distances = [self.euclidean_distance(self.pose.position, self.waypoints[closest_waypoint].pose.pose.position)]
        distances += self.distance_list(self.waypoints, closest_waypoint, end)
        for i in range(1, len(distances)):
            distances[i] = distances[i] + distances[i - 1]

        waypoints = []
        for idx in range(closest_waypoint, end + 1):
            dist = distances[idx - closest_waypoint]
            velocity = math.sqrt(init_vel**2 + 2 * self.accel_limit * dist)
            if velocity > self.speed_limit:
                velocity = self.speed_limit
            waypoint = Waypoint()
            waypoint.pose = self.waypoints[idx].pose
            waypoint.twist.twist.linear.x = velocity
            waypoints.append(waypoint)
        return self.build_lane(waypoints)

    def slowdown_waypoints(self, stop_idx):
        linear_vel = self.current_velocity.linear
        init_vel = math.sqrt(linear_vel.x ** 2 +  linear_vel.y ** 2)

        closest_waypoint = self.get_closest_waypoint_idx()
        distance_to_closest_waypoint = self.euclidean_distance(
            self.pose.position, self.waypoints[closest_waypoint].pose.pose.position
        )
        distance_to_tl = distance_to_closest_waypoint + self.distance(self.waypoints, closest_waypoint, stop_idx)

        decel = (init_vel ** 2)/(2 * distance_to_tl)

        end = min(closest_waypoint + LOOKAHEAD_WPS, len(self.waypoints) - 1)
        distances = [distance_to_closest_waypoint]
        distances += self.distance_list(self.waypoints, closest_waypoint, end)
        for i in range(1, len(distances)):
            distances[i] = distances[i] + distances[i - 1]

        waypoints = []
        for idx in range(closest_waypoint, end + 1):
            dist = distances[idx - closest_waypoint]
            velocity = init_vel**2 - 2 * decel * dist
            if velocity < 0.1:
                velocity = 0.0
            velocity = math.sqrt(velocity)
            waypoint = Waypoint()
            waypoint.pose = self.waypoints[idx].pose
            waypoint.twist.twist.linear.x = velocity
            waypoints.append(waypoint)
        return self.build_lane(waypoints)

    def build_lane(self, waypoints):
        lane = Lane()
        lane.header = self.original_waypoints.header
        lane.waypoints = waypoints
        return lane

    def get_closest_waypoint_idx(self):
        x_coordinate = self.pose.position.x
        y_coordinate = self.pose.position.y
        closest_idx = self.waypoint_tree.query([x_coordinate, y_coordinate], 1)[1]

        # Check if closest is ahead or behind vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx-1]

        # Equation for hyperplane through closest_coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x_coordinate, y_coordinate])

        val = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx

    def pose_cb(self, msg):
        self.pose = msg.pose

    def waypoints_cb(self, lane):
        self.original_waypoints = lane
        self.waypoints = lane.waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y]
                                 for waypoint in self.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.traffic_waypoint = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def current_velocity_cb(self, msg):
        self.current_velocity = msg.twist

    def euclidean_distance(self, point_a, point_b):
        return math.sqrt((point_a.x-point_b.x)**2 + (point_a.y-point_b.y)**2 + (point_a.z-point_b.z)**2)

    def distance_list(self, waypoints, wp1, wp2):
        distances = []
        for i in range(wp1, wp2):
            distances.append(self.euclidean_distance(
                waypoints[i].pose.pose.position,
                waypoints[i + 1].pose.pose.position
            ))
        for i in range(1, len(distances)):
            distances[i] = distances[i] + distances[i - 1]
        return distances

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        for i in range(wp1, wp2+1):
            dist += self.euclidean_distance(
                waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position
            )
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
