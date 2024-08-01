import math

#Work on the completing track strategy, speed and stay as much as possible in the middle
#Implement the logic that in sharp corners only three wheels can veer off but not all of them 
#Track completion - 100% without veering off the road is the goal
#Affirm the final speed to be 1.5 and 4 as min and max respectively

class PARAMS:
    prev_speed = None
    prev_steering_angle = None 
    prev_steps = None
    prev_direction_diff = None
    prev_normalized_distance_from_route = None
    unpardonable_action = False
    intermediate_progress = [0] * 11

def reward_function(params):

    #Constants
    SPEED_INCREASE_BONUS_DEFAULT = 2
    OFF_TRACK_PENALTY_THRESHOLD = 0.5
    OFF_TRACK_PENALTY_MIN = 0.1
    HEADING_DECREASE_BONUS_MAX = 10
    DIRECTION_DIFF_THRESHOLD_1 = 10
    DIRECTION_DIFF_THRESHOLD_2 = 5
    STEERING_ANGLE_MAINTAIN_BONUS_MULTIPLIER = 2
    
    # Read input parameters
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
    wheels_on_track = params.get('wheels_on_track', 4)  # Assuming 4 wheels by default

    # Define a safety limit for reward to avoid excessive values
    MAX_REWARD = 1e3

    # Calculate the next waypoint
    next_point = waypoints[closest_waypoints[1]]

    # Reinitialize previous parameters if it is a new episode
    if PARAMS.prev_steps is None or steps < PARAMS.prev_steps:
        PARAMS.prev_speed = None
        PARAMS.prev_steering_angle = None
        PARAMS.prev_direction_diff = None
        PARAMS.prev_normalized_distance_from_route = None
        PARAMS.intermediate_progress = [0] * 11

    # Check if the speed has decreased
    has_speed_dropped = PARAMS.prev_speed is not None and PARAMS.prev_speed > speed

    # Define the minimum and maximum speeds
    min_speed = 1.5
    max_speed = 4.0

    # Calculate the speed reward
    speed_reward = calculate_speed_reward(speed, min_speed, max_speed)

    # Penalize slowing down without a valid reason on straight roads
    speed_maintain_bonus = 1
    if has_speed_dropped and not is_turn_upcoming:
        speed_maintain_bonus = min(speed / max(PARAMS.prev_speed, min_speed), 1)

    # Check if the speed has increased - provide additional rewards
    has_speed_increased = PARAMS.prev_speed is not None and PARAMS.prev_speed < speed
    speed_increase_bonus = SPEED_INCREASE_BONUS_DEFAULT
    if has_speed_increased and not is_turn_upcoming:
        speed_increase_bonus = max(speed / max(PARAMS.prev_speed, min_speed), 1)

    # Penalize moving off track
    off_track_penalty = 1 - abs(normalized_distance_from_route)
    if off_track_penalty < OFF_TRACK_PENALTY_THRESHOLD:
        off_track_penalty = OFF_TRACK_PENALTY_MIN

    # Penalize making the heading direction worse
    heading_bonus = 0
    direction_diff = calculate_direction_diff(heading, vehicle_x, vehicle_y, next_point)
    if PARAMS.prev_direction_diff is not None and is_heading_in_right_direction:
        if abs(PARAMS.prev_direction_diff / direction_diff) > 1:
            heading_bonus = min(HEADING_DECREASE_BONUS_MAX, abs(PARAMS.prev_direction_diff / direction_diff))

    # Check if the steering angle has changed
    has_steering_angle_changed = PARAMS.prev_steering_angle is not None and not math.isclose(PARAMS.prev_steering_angle, steering_angle)

    # Not changing the steering angle is good if heading in the right direction
    steering_angle_maintain_bonus = 1
    if is_heading_in_right_direction and not has_steering_angle_changed:
        if abs(direction_diff) < DIRECTION_DIFF_THRESHOLD_1:
            steering_angle_maintain_bonus *= STEERING_ANGLE_MAINTAIN_BONUS_MULTIPLIER
        if abs(direction_diff) < DIRECTION_DIFF_THRESHOLD_2:
            steering_angle_maintain_bonus *= STEERING_ANGLE_MAINTAIN_BONUS_MULTIPLIER
        if PARAMS.prev_direction_diff is not None and abs(PARAMS.prev_direction_diff) > abs(direction_diff):
            steering_angle_maintain_bonus *= STEERING_ANGLE_MAINTAIN_BONUS_MULTIPLIER

    # Reward reducing distance to the race line
    distance_reduction_bonus = 1
    if PARAMS.prev_normalized_distance_from_route is not None and PARAMS.prev_normalized_distance_from_route > normalized_distance_from_route:
        if abs(normalized_distance_from_route) > 0:
            distance_reduction_bonus = min(abs(PARAMS.prev_normalized_distance_from_route / normalized_distance_from_route), 2)

    # Before returning reward, update the variables
    PARAMS.prev_speed = speed
    PARAMS.prev_steering_angle = steering_angle
    PARAMS.prev_direction_diff = direction_diff
    PARAMS.prev_steps = steps
    PARAMS.prev_normalized_distance_from_route = normalized_distance_from_route

    # Calculate rewards
    heading_reward = calculate_heading_reward(heading, vehicle_x, vehicle_y, next_point)
    distance_reward = calculate_distance_reward(bearing, normalized_car_distance_from_route, normalized_route_distance_from_inner_border, normalized_route_distance_from_outer_border)
    speed_reward = speed_reward

    # Heading component of reward
    HC = 10 * heading_reward * steering_angle_maintain_bonus
    # Distance component of reward
    DC = 10 * distance_reward * distance_reduction_bonus
    # Speed component of reward
    SC = 10 * speed_reward * speed_maintain_bonus * speed_increase_bonus
    # Immediate component of reward
    IC = (HC + DC + SC) ** 2 + (HC * DC * SC)
    # If an unpardonable action is taken, then the immediate reward is 0
    if PARAMS.unpardonable_action:
        IC = 1e-3
    # Long term component of reward
    intermediate_progress_bonus = calculate_intermediate_progress_bonus(progress, steps)
    LC = curve_bonus + intermediate_progress_bonus + straight_section_bonus

    # Incentive for finishing the track
    if progress == 100:
        LC += 500  # Large bonus for finishing the track

    # Sharp corner logic: allow up to three wheels off track
    if is_turn_upcoming:
        if wheels_on_track < 1:
            total_reward = 1e-3  # Heavy penalty if all wheels are off track
        elif wheels_on_track < 4:
            total_reward *= 0.5  # Moderate penalty if up to three wheels are off track
    else:
        if not all_wheels_on_track:
            total_reward = 1e-3  # Heavy penalty if not all wheels are on track

    total_reward = max(IC + LC, 1e-3) * off_track_penalty

    # Apply a cap to the reward to avoid excessive values
    total_reward = min(total_reward, MAX_REWARD)

    return total_reward

def calculate_speed_reward(speed, min_speed, max_speed):
    
    if speed < min_speed:
        return 0.1  # Penalize low speeds
    elif speed > max_speed:
        return 1.0  # Maximum reward for speeds above the maximum
    else:
        return (speed - min_speed) / (max_speed - min_speed)  # Reward higher speeds proportionally



def calculate_distance_reward(bearing, normalized_car_distance_from_route, normalized_route_distance_from_inner_border, normalized_route_distance_from_outer_border):
    distance_reward = 0
    if "center" in bearing:  # i.e., on the route line
        distance_from_route = 0
        distance_reward = 1
    elif "right" in bearing:  # i.e., on right side of the route line
        sigma = abs(normalized_route_distance_from_inner_border / 4)
        distance_reward = math.exp(-0.5 * abs(normalized_car_distance_from_route) ** 2 / sigma ** 2)
    elif "left" in bearing:  # i.e., on left side of the route line
        sigma = abs(normalized_route_distance_from_outer_border / 4)
        distance_reward = math.exp(-0.5 * abs(normalized_car_distance_from_route) ** 2 / sigma ** 2)
    return distance_reward

def calculate_intermediate_progress_bonus(progress, steps):
    progress_reward = 10 * progress / steps
    if steps <= 5:
        progress_reward = 1  # Ignore progress in the first 5 steps

    intermediate_progress_bonus = 0
    pi = int(progress // 10)
    if pi != 0 and PARAMS.intermediate_progress[pi] == 0:
        if pi == 10:  # 100% track completion
            intermediate_progress_bonus = progress_reward ** 14
        else:
            intermediate_progress_bonus = progress_reward ** (5 + 0.75 * pi)
        PARAMS.intermediate_progress[pi] = intermediate_progress_bonus

    return intermediate_progress_bonus

def calculate_direction_diff(heading, vehicle_x, vehicle_y, next_point):
    next_point_x = next_point[0]
    next_point_y = next_point[1]

    # Calculate the direction in radians, arctan2(dy, dx), the result is (-pi, pi) in radians between target and current vehicle position
    route_direction = math.atan2(next_point_y - vehicle_y, next_point_x - vehicle_x)
    # Convert to degrees
    route_direction = math.degrees(route_direction)
    # Calculate the difference between the track direction and the heading direction of the car
    direction_diff = route_direction - heading
    return direction_diff

def calculate_heading_reward(heading, vehicle_x, vehicle_y, next_point):
    next_point_x = next_point[0]
    next_point_y = next_point[1]

    # Calculate the direction in radians, arctan2(dy, dx), the result is (-pi, pi) in radians between target and current vehicle position
    route_direction = math.atan2(next_point_y - vehicle_y, next_point_x - vehicle_x)
    # Convert to degrees
    route_direction = math.degrees(route_direction)
    # Calculate the difference between the track direction and the heading direction of the car
    direction_diff = route_direction - heading
    # Check that the direction_diff is in valid range
    # Then compute the heading reward
    heading_reward = math.cos(abs(direction_diff) * (math.pi / 180)) ** 10
    if abs(direction_diff) <= 20:
        heading_reward = math.cos(abs(direction_diff) * (math.pi / 180)) ** 4

    return heading_reward