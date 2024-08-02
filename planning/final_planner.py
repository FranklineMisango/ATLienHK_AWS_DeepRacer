import math

# Constants
SHARP_CORNER_THRESHOLD = 30
MIN_SPEED = 1.5
MAX_SPEED = 4.0
OFF_TRACK_PENALTY_THRESHOLD = 0.5
OFF_TRACK_PENALTY_MIN = 0.1
HEADING_DECREASE_BONUS_MAX = 10
DIRECTION_DIFF_THRESHOLD_1 = 10
DIRECTION_DIFF_THRESHOLD_2 = 5
STEERING_ANGLE_MAINTAIN_BONUS_MULTIPLIER = 2
SPEED_INCREASE_BONUS_DEFAULT = 2
MAX_REWARD = 1e3

class PARAMS:
    prev_speed = None
    prev_steering_angle = None
    prev_steps = None
    prev_direction_diff = None
    prev_normalized_distance_from_route = None
    intermediate_progress = [0] * 11
    unpardonable_action = False

class Vehicle:
    def __init__(self):
        self.speed = 0
        self.position = (0, 0)
        self.direction = 0  # Angle in degrees
        self.track = []  # List of track points

    def detect_sharp_corner(self):
        current_index = self.track.index(self.position)
        if current_index < len(self.track) - 1:
            next_position = self.track[current_index + 1]
            curvature = self.calculate_curvature(self.position, next_position)
            return curvature > SHARP_CORNER_THRESHOLD
        return False

    def calculate_curvature(self, pos1, pos2):
        return abs(pos2[0] - pos1[0]) + abs(pos2[1] - pos1[1])

    def calculate_turning_angle(self, current_direction, desired_direction):
        angle_diff = desired_direction - current_direction
        if angle_diff > 180:
            angle_diff -= 360
        elif angle_diff < -180:
            angle_diff += 360
        return max(min(angle_diff, 30), -30)

    def adjust_speed_for_corner(self):
        if self.detect_sharp_corner():
            self.speed = max(self.speed - 10, 0)  # Slow down
        else:
            self.speed = min(self.speed + 5, 100)  # Speed up

    def update_position(self):
        current_index = self.track.index(self.position)
        if current_index < len(self.track) - 1:
            next_position = self.track[current_index + 1]
            desired_direction = math.degrees(math.atan2(next_position[1] - self.position[1], next_position[0] - self.position[0]))
            turning_angle = self.calculate_turning_angle(self.direction, desired_direction)
            self.direction += turning_angle
            self.position = next_position

    def run(self):
        while True:
            self.adjust_speed_for_corner()
            self.update_position()

def calculate_turning_angle(current_position, next_position):
    delta_x = next_position[0] - current_position[0]
    delta_y = next_position[1] - current_position[1]
    return math.degrees(math.atan2(delta_y, delta_x))

def adjust_speed_for_corner(current_speed, turning_angle, consecutive_sharp_turns):
    if consecutive_sharp_turns:
        return max(current_speed * 0.3, 1)
    return max(current_speed * 0.5, 1) if abs(turning_angle) > 20 else current_speed

def detect_consecutive_sharp_turns(waypoints, current_index, threshold=20):
    for i in range(current_index, min(current_index + 3, len(waypoints) - 1)):
        angle = calculate_turning_angle(waypoints[i], waypoints[i + 1])
        if abs(angle) > threshold:
            return True
    return False

def calculate_speed_reward(speed):
    if speed < MIN_SPEED:
        return 0.1
    if speed > MAX_SPEED:
        return 1.0
    return (speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)

def calculate_distance_reward(bearing, normalized_car_distance_from_route, normalized_route_distance_from_inner_border, normalized_route_distance_from_outer_border):
    if "center" in bearing:
        return 1
    sigma = abs(normalized_route_distance_from_inner_border / 4) if "right" in bearing else abs(normalized_route_distance_from_outer_border / 4)
    return math.exp(-0.5 * abs(normalized_car_distance_from_route) ** 2 / sigma ** 2)

def calculate_intermediate_progress_bonus(progress, steps):
    if steps <= 5:
        return 0
    progress_reward = 10 * progress / steps
    pi = int(progress // 10)
    if pi != 0 and PARAMS.intermediate_progress[pi] == 0:
        bonus = progress_reward ** (14 if pi == 10 else (5 + 0.75 * pi))
        PARAMS.intermediate_progress[pi] = bonus
        return bonus
    return 0

def calculate_direction_diff(heading, vehicle_x, vehicle_y, next_point):
    route_direction = math.degrees(math.atan2(next_point[1] - vehicle_y, next_point[0] - vehicle_x))
    return route_direction - heading

def calculate_heading_reward(heading, vehicle_x, vehicle_y, next_point):
    direction_diff = calculate_direction_diff(heading, vehicle_x, vehicle_y, next_point)
    heading_reward = math.cos(abs(direction_diff) * (math.pi / 180)) ** 10
    return heading_reward if abs(direction_diff) <= 20 else heading_reward ** 0.4

def reward_function(params):
    heading = params['heading']
    distance_from_center = params['distance_from_center']
    steps = params['steps']
    steering_angle = params['steering_angle']
    speed = params['speed']
    progress = params.get('progress', 0)
    bearing = params.get('bearing', "")
    normalized_car_distance_from_route = params.get('normalized_car_distance_from_route', 0)
    normalized_route_distance_from_inner_border = params.get('normalized_route_distance_from_inner_border', 0)
    normalized_route_distance_from_outer_border = params.get('normalized_route_distance_from_outer_border', 0)
    vehicle_x = params['x']
    vehicle_y = params['y']
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    is_turn_upcoming = params.get('is_turn_upcoming', False)
    is_heading_in_right_direction = params.get('is_heading_in_right_direction', False)
    normalized_distance_from_route = params.get('normalized_distance_from_route', 0)
    curve_bonus = params.get('curve_bonus', 0)
    straight_section_bonus = params.get('straight_section_bonus', 0)
    all_wheels_on_track = params['all_wheels_on_track']
    wheels_on_track = params.get('wheels_on_track', 4)

    next_point = waypoints[closest_waypoints[1]]

    if PARAMS.prev_steps is None or steps < PARAMS.prev_steps:
        PARAMS.prev_speed = None
        PARAMS.prev_steering_angle = None
        PARAMS.prev_direction_diff = None
        PARAMS.prev_normalized_distance_from_route = None
        PARAMS.unpardonable_action = False

    has_speed_dropped = PARAMS.prev_speed is not None and PARAMS.prev_speed > speed
    speed_reward = calculate_speed_reward(speed)
    speed_maintain_bonus = 1 if not has_speed_dropped or is_turn_upcoming else min(speed / max(PARAMS.prev_speed, MIN_SPEED), 1)
    speed_increase_bonus = SPEED_INCREASE_BONUS_DEFAULT if has_speed_dropped and not is_turn_upcoming else max(speed / max(PARAMS.prev_speed, MIN_SPEED), 1)

    off_track_penalty = 1 - abs(normalized_distance_from_route)
    off_track_penalty = max(off_track_penalty, OFF_TRACK_PENALTY_MIN) if off_track_penalty < OFF_TRACK_PENALTY_THRESHOLD else off_track_penalty

    direction_diff = calculate_direction_diff(heading, vehicle_x, vehicle_y, next_point)
    heading_bonus = 0
    if PARAMS.prev_direction_diff is not None and is_heading_in_right_direction:
        if abs(PARAMS.prev_direction_diff / direction_diff) > 1:
            heading_bonus = min(HEADING_DECREASE_BONUS_MAX, abs(PARAMS.prev_direction_diff / direction_diff))

    steering_angle_maintain_bonus = 1
    if is_heading_in_right_direction and PARAMS.prev_steering_angle is not None:
        has_steering_angle_changed = not math.isclose(PARAMS.prev_steering_angle, steering_angle)
        if not has_steering_angle_changed:
            if abs(direction_diff) < DIRECTION_DIFF_THRESHOLD_1:
                steering_angle_maintain_bonus *= STEERING_ANGLE_MAINTAIN_BONUS_MULTIPLIER
            if abs(direction_diff) < DIRECTION_DIFF_THRESHOLD_2:
                steering_angle_maintain_bonus *= STEERING_ANGLE_MAINTAIN_BONUS_MULTIPLIER
            if PARAMS.prev_direction_diff is not None and abs(PARAMS.prev_direction_diff) > abs(direction_diff):
                steering_angle_maintain_bonus *= STEERING_ANGLE_MAINTAIN_BONUS_MULTIPLIER

    distance_reduction_bonus = 1
    if PARAMS.prev_normalized_distance_from_route is not None and PARAMS.prev_normalized_distance_from_route > normalized_distance_from_route:
        distance_reduction_bonus = min(abs(PARAMS.prev_normalized_distance_from_route / normalized_distance_from_route), 2)

    PARAMS.prev_speed = speed
    PARAMS.prev_steering_angle = steering_angle
    PARAMS.prev_direction_diff = direction_diff
    PARAMS.prev_steps = steps
    PARAMS.prev_normalized_distance_from_route = normalized_distance_from_route

    heading_reward = calculate_heading_reward(heading, vehicle_x, vehicle_y, next_point)
    distance_reward = calculate_distance_reward(bearing, normalized_car_distance_from_route, normalized_route_distance_from_inner_border, normalized_route_distance_from_outer_border)

    HC = 10 * heading_reward * steering_angle_maintain_bonus
    DC = 10 * distance_reward * distance_reduction_bonus
    SC = 10 * speed_reward * speed_maintain_bonus * speed_increase_bonus
    IC = (HC + DC + SC) ** 2 + (HC * DC * SC)

    if PARAMS.unpardonable_action:
        IC = 1e-3

    intermediate_progress_bonus = calculate_intermediate_progress_bonus(progress, steps)
    LC = curve_bonus + intermediate_progress_bonus + straight_section_bonus

    if progress == 100:
        LC += 500

    if is_turn_upcoming:
        if wheels_on_track < 1:
            total_reward = 1e-3
        elif wheels_on_track < 4:
            IC *= 0.5
    else:
        if not all_wheels_on_track:
            total_reward = 1e-3

    total_reward = max(IC + LC, 1e-3) * off_track_penalty
    return min(total_reward, MAX_REWARD)
