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
WPS_CORRECTION = 3 # wps

class WaypointUpdater(object):

    def __init__(self):
        rospy.init_node('waypoint_updater')

        # Parameters
        self.min_decel = rospy.get_param('~/twist_controller/decel_limit', -1.0)
        self.speed_limit = rospy.get_param('/waypoint_loader/velocity')
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
        self.traffic_waypoint_idx = -1
        self.current_twist = None

        # Node state compute on each iteration
        self.current_velocity2 = None
        self.closest_waypoint_idx = None
        self.dist_to_closest_waypoint = None

        # Constants for speeding up execution
        self.decelx2 = 2.0 * self.min_decel

        # Traffic light variables
        self.tl_changed = True
        self.able_to_stop = True
        self.last_traffic_waypoint_idx = -1

        self.loop()

    def loop(self):

        rate = rospy.Rate(15)

        while not rospy.is_shutdown():

            # Compute action
            action, context = self.desired_action()

            if action:
                # Compute new waypoints
                lane = self.build_final_waypoints(action, context)
                self.final_waypoints_pub.publish(lane)

            rate.sleep()

    def desired_action(self):
        
        if not (self.pose and self.waypoint_tree and self.current_twist):
            return None, {}
                 
        # Red light detected ahead
        if self.closest_waypoint_idx <= self.traffic_waypoint_idx:

            dist_to_tl = self.distance(self.traffic_waypoint_idx)

            # Check if we are able to stop before the traffic light (green might have turned yellow as we cross).
            if self.tl_changed == True:
                self.tl_changed = False

                # Car break distance at maximum deceleration
                break_distance = -self.current_velocity2/self.decelx2

                if dist_to_tl > break_distance: 
                    self.able_to_stop = True
                else:
                    self.able_to_stop = False

            if self.able_to_stop == True:
                return 'SLOWDOWN', {'dist': dist_to_tl}

        return 'BASE', {}

    def build_final_waypoints(self, action, context):
        if action == 'SLOWDOWN':
            return self.slowdown_waypoints(**context)
        return self.base_waypoints(**context)

    def slowdown_waypoints(self, dist):

        dist_brake = dist - self.dist_to_closest_waypoint
        waypoints = []
        end_wp = min(self.closest_waypoint_idx+LOOKAHEAD_WPS, self.n_waypoints)

        for idx in range(self.closest_waypoint_idx, end_wp):

            # All waypoints after the traffic light have zero velocity
            if idx > self.traffic_waypoint_idx:

                velocity = 0.0

            else:

                # Get the set velocity for the current waypoint
                wp_velocity = self.original_waypoints.waypoints[idx].twist.twist.linear.x

                # Calculate velocity at the next waypoint
                br_velocity = math.sqrt(max(-self.decelx2 * dist_brake, 0.0))
                dist_brake = dist_brake - self.euc_distances[idx]

                velocity = min(wp_velocity, br_velocity)

                if velocity < 0.1:
                    velocity = 0.0

                # If the car stopped a few waypoints before a red traffic light, do not attempt to go any further
                if self.current_twist.linear.x < 0.2 and (self.traffic_waypoint_idx - self.closest_waypoint_idx) <= 2:
                    velocity = 0.0

#            if (idx == self.closest_waypoint_idx):                
#                rospy.loginfo("WUP: %.4f; %.4f; %.4f; %.4f; %.4f; %.4f; %.4f; 0", self.current_twist.linear.x, velocity, wp_velocity, br_velocity, self.traffic_waypoint_idx, self.pose.position.x, self.pose.position.y)

            # Create the waypoint
            waypoint = Waypoint()
            waypoint.pose = self.waypoints[idx].pose
            waypoint.twist.twist.linear.x = velocity
            waypoints.append(waypoint)

        return self.build_lane(waypoints)

    def base_waypoints(self):

        waypoints = []
        end_wp = min(self.closest_waypoint_idx+LOOKAHEAD_WPS, self.n_waypoints)

        for idx in range(self.closest_waypoint_idx, end_wp):

            # Get the set velocity for the current waypoint
            wp_velocity = self.original_waypoints.waypoints[idx].twist.twist.linear.x
            velocity = wp_velocity

#            if (idx == self.closest_waypoint_idx):
#                rospy.loginfo("WUP: %.4f; %.4f; %.4f; %.4f; %.4f; %.4f; %.4f; 1", self.current_twist.linear.x, velocity, wp_velocity, -2, self.traffic_waypoint_idx, self.pose.position.x, self.pose.position.y)

            # Create the waypoint
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

    def pose_cb(self, msg):
        self.pose = msg.pose

        if self.waypoint_tree:
            # Closest waypoint
            self.closest_waypoint_idx = self.get_closest_waypoint_idx()
           
            # Distance to closest waypoint
            self.dist_to_closest_waypoint = self.euclidean_distance(
                self.pose.position,
                self.waypoints[self.closest_waypoint_idx].pose.pose.position
            )

    def waypoints_cb(self, lane):

        # Get the map waypoints. Function is executed only once
        self.original_waypoints = lane
        self.waypoints = lane.waypoints
        self.n_waypoints = len(self.waypoints)

        if not self.waypoints_2d:

            # Calculate the KDTree
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y]
                                 for waypoint in self.waypoints]
            self.waypoint_tree = cKDTree(self.waypoints_2d, leafsize=1)

            # Euclidean distances from each waypoint to the next
            self.euc_distances = np.empty(self.n_waypoints, dtype=float)
            for i in range(self.n_waypoints-1):
                self.euc_distances[i] = self.euclidean_distance(
                    self.waypoints[i].pose.pose.position,
                    self.waypoints[i + 1].pose.pose.position)
            
            # Distance from the last waypoint to first one
            self.euc_distances[self.n_waypoints-1] = self.euclidean_distance(
                self.waypoints[self.n_waypoints-1].pose.pose.position,
                self.waypoints[0].pose.pose.position)

    def traffic_cb(self, msg):
        idx = msg.data

        # Subtract some waypoints so that the car stops behind the stop line
        if idx >= WPS_CORRECTION:
            idx = idx - WPS_CORRECTION

        self.traffic_waypoint_idx = idx

        # Check if the traffic light has turned yellow
        if self.last_traffic_waypoint_idx == -1 and self.traffic_waypoint_idx >= 0:
            self.tl_changed = True

        self.last_traffic_waypoint_idx = self.traffic_waypoint_idx    

    def current_twist_cb(self, msg):
        self.current_twist = msg.twist

        # Car velocity
        linear_vel = self.current_twist.linear
        self.current_velocity2 = linear_vel.x ** 2# +  linear_vel.y ** 2

    def get_closest_waypoint_idx(self):
        x = self.pose.position.x
        y = self.pose.position.y
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        pos_vect = np.array([x, y])

        # Loop in order for the closest waypoint to be in front of the car
        for _ in range(3):

            # Check if closest is ahead or behind vehicle
            closest_coord = self.waypoints_2d[closest_idx]
            prev_coord    = self.waypoints_2d[closest_idx-1]
		
            # Equation for hyperplane through closest_coords
            cl_vect = np.array(closest_coord)
            prev_vect = np.array(prev_coord)
		
            val = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)

            if val > 0 and closest_idx < self.n_waypoints-1:
                closest_idx = closest_idx + 1
            else:
                return closest_idx

        return closest_idx

    def euclidean_distance(self, point_a, point_b):
        return math.sqrt(
            (point_a.x-point_b.x)**2 + (point_a.y-point_b.y)**2 + (point_a.z-point_b.z)**2
        )

    def distance(self, wp2):
        return self.dist_to_closest_waypoint + self.euc_distances[self.closest_waypoint_idx:wp2].sum()

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')

