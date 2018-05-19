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

LOOKAHEAD_WPS = 150 # Number of waypoints we will publish. You can change this number
ACC_FACTOR = 1.0
DECEL_FACTOR = 1.0
MAX_VEL_FACTOR = 1.0

class WaypointUpdater(object):

    def __init__(self):
        rospy.init_node('waypoint_updater')

        # Parameters
        self.accel_limit = rospy.get_param('~/twist_controller/accel_limit', 1.0) * ACC_FACTOR
        self.min_decel = rospy.get_param('~/twist_controller/decel_limit', -1.0) * DECEL_FACTOR
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
        self.traffic_waypoint_idx = -1
        self.current_twist = None

        # Node state compute on each iteration
        self.current_velocity2 = None
        self.closest_waypoint_idx = None
        self.dist_to_closest_waypoint = None

        # Constants for speeding up execution
        self.decelx2 = -2.0 * self.min_decel
        self.accelx2 = 2.0 * self.accel_limit

        self.loop()

    def loop(self):
        rate = rospy.Rate(10)
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
            self.current_velocity2 = linear_vel.x ** 2# +  linear_vel.y ** 2

        if self.pose and self.waypoint_tree:
            # Closest waypoint
            self.closest_waypoint_idx = self.get_closest_waypoint_idx()
           
            # Distance to closest waypoint
            self.dist_to_closest_waypoint = self.euclidean_distance(
                self.pose.position,
                self.waypoints[self.closest_waypoint_idx].pose.pose.position
            )

    def desired_action(self):
        
        if not (self.pose and self.waypoint_tree and self.current_twist):
            return None, {}

        # Red light detected
        if (self.traffic_waypoint_idx > -1):

            if (self.closest_waypoint_idx + LOOKAHEAD_WPS > self.n_waypoints-1):
                traffic_idx = self.traffic_waypoint_idx + self.n_waypoints
            else:
                traffic_idx = self.traffic_waypoint_idx            

            # Check if the traffic light is within the next LOOKAHEAD_WPS number of waypoints
            if (traffic_idx >=  self.closest_waypoint_idx and
                traffic_idx <= (self.closest_waypoint_idx + LOOKAHEAD_WPS)):

                dist_to_tl = self.distance(self.traffic_waypoint_idx)
                #rospy.loginfo("WUP: tl=%s, dist=%s, vel=%s", self.traffic_waypoint_idx, dist_to_tl, self.current_twist.linear.x)

                return 'SLOWDOWN', {'dist': dist_to_tl}

        #rospy.loginfo("WUP: accelerate")
        return 'ACCELERATE', {}

    def build_final_waypoints(self, action, context):
        if action == 'SLOWDOWN':
            return self.slowdown_waypoints(**context)
        return self.accelerate_waypoints(**context)

    def slowdown_waypoints(self, dist):

        flag_stop = False
        velocity2 = self.current_velocity2
        waypoints = []

        for i in range(LOOKAHEAD_WPS):

            idx = (self.closest_waypoint_idx + i) % self.n_waypoints

            # All waypoints after the traffic light have zero velocity
            if (idx == self.traffic_waypoint_idx or flag_stop == True):

                flag_stop = True
                velocity = 0.0

            else:

                # Calculate velocity at the next waypoint
                velocity = math.sqrt(max(self.decelx2 * dist, 0.0))
                dist = dist - self.euc_distances[idx]

                velocity_max = math.sqrt(max(velocity2 + self.accelx2 * self.euc_distances[idx], 0.0))

                velocity = min(max(self.speed_limit, velocity_max), velocity)

                if velocity < 0.1:
                    velocity = 0.0
                velocity2 = velocity*velocity

            # Create the waypoint
            waypoint = Waypoint()
            waypoint.pose = self.waypoints[idx].pose
            waypoint.twist.twist.linear.x = velocity     
            waypoints.append(waypoint)

        return self.build_lane(waypoints)

    def accelerate_waypoints(self):

        dist = self.dist_to_closest_waypoint
        velocity2 = self.current_velocity2
        waypoints = []

        for i in range(LOOKAHEAD_WPS):

            idx = (self.closest_waypoint_idx + i) % self.n_waypoints

            # Calculate velocity at the next waypoint
            velocity = math.sqrt(max(velocity2 + self.accelx2 * dist, 0.0))
            dist = self.euc_distances[idx]

            # Do not go faster than the speed limit
            if velocity > self.speed_limit:
                velocity = self.speed_limit
            velocity2 = velocity*velocity

            #rospy.loginfo("WUP: acc cvel=%s, vel=%s, limit=%s", self.current_twist.linear.x, velocity, self.speed_limit)
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
        self.traffic_waypoint_idx = msg.data

        # Subtract some waypoints so that the car stops behind the stop line
        if (self.traffic_waypoint_idx >= 0):
            self.traffic_waypoint_idx = self.traffic_waypoint_idx - 3
            # Handle underflow situation
            if (self.traffic_waypoint_idx < 0):
                self.traffic_waypoint_idx = self.traffic_waypoint_idx + self.n_waypoints

    def current_twist_cb(self, msg):
        self.current_twist = msg.twist

    def get_closest_waypoint_idx(self):
        x = self.pose.position.x
        y = self.pose.position.y
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]

        #while True:
        for i in range(3):

            # Check if closest is ahead or behind vehicle
            closest_coord = self.waypoints_2d[closest_idx]
            prev_coord    = self.waypoints_2d[closest_idx-1]
		
            # Equation for hyperplane through closest_coords
            cl_vect = np.array(closest_coord)
            prev_vect = np.array(prev_coord)
            pos_vect = np.array([x, y])
		
            val = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)

            if val > 0:
                closest_idx = (closest_idx + 1) % self.n_waypoints
            else:
                break;

        return closest_idx

    def euclidean_distance(self, point_a, point_b):
        return math.sqrt(
            (point_a.x-point_b.x)**2 + (point_a.y-point_b.y)**2 + (point_a.z-point_b.z)**2
        )

    def distance(self, wp2):
        idx = self.closest_waypoint_idx
        dist = self.dist_to_closest_waypoint
        while (idx != wp2):
            dist = dist + self.euc_distances[idx]
            idx = (idx + 1) % self.n_waypoints
        return dist

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')

