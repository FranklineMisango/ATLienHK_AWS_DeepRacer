import math

##action space 
{
  "action_space": [
    {"steering_angle": -30.0, "speed": 0.5},
    {"steering_angle": -27.5, "speed": 0.5},
    {"steering_angle": -25.0, "speed": 0.5},
    {"steering_angle": -22.5, "speed": 0.5},
    {"steering_angle": -20.0, "speed": 0.5},
    {"steering_angle": -17.5, "speed": 0.5},
    {"steering_angle": -15.0, "speed": 0.5},
    {"steering_angle": -12.5, "speed": 0.5},
    {"steering_angle": -10.0, "speed": 0.5},
    {"steering_angle": -7.5, "speed": 0.5},
    {"steering_angle": -5.0, "speed": 0.5},
    {"steering_angle": -2.5, "speed": 0.5},
    {"steering_angle": 0.0, "speed": 0.5},
    {"steering_angle": 2.5, "speed": 0.5},
    {"steering_angle": 5.0, "speed": 0.5},
    {"steering_angle": 7.5, "speed": 0.5},
    {"steering_angle": 10.0, "speed": 0.5},
    {"steering_angle": 12.5, "speed": 0.5},
    {"steering_angle": 15.0, "speed": 0.5},
    {"steering_angle": 17.5, "speed": 0.5},
    {"steering_angle": 20.0, "speed": 0.5},
    {"steering_angle": 22.5, "speed": 0.5},
    {"steering_angle": 25.0, "speed": 0.5},
    {"steering_angle": 27.5, "speed": 0.5},
    {"steering_angle": 30.0, "speed": 0.5},
    {"steering_angle": -30.0, "speed": 1.0},
    {"steering_angle": -27.5, "speed": 1.0},
    {"steering_angle": -25.0, "speed": 1.0},
    {"steering_angle": -22.5, "speed": 1.0},
    {"steering_angle": -20.0, "speed": 1.0},
    {"steering_angle": -17.5, "speed": 1.0},
    {"steering_angle": -15.0, "speed": 1.0},
    {"steering_angle": -12.5, "speed": 1.0},
    {"steering_angle": -10.0, "speed": 1.0},
    {"steering_angle": -7.5, "speed": 1.0},
    {"steering_angle": -5.0, "speed": 1.0},
    {"steering_angle": -2.5, "speed": 1.0},
    {"steering_angle": 0.0, "speed": 1.0},
    {"steering_angle": 2.5, "speed": 1.0},
    {"steering_angle": 5.0, "speed": 1.0},
    {"steering_angle": 7.5, "speed": 1.0},
    {"steering_angle": 10.0, "speed": 1.0},
    {"steering_angle": 12.5, "speed": 1.0},
    {"steering_angle": 15.0, "speed": 1.0},
    {"steering_angle": 17.5, "speed": 1.0},
    {"steering_angle": 20.0, "speed": 1.0},
    {"steering_angle": 22.5, "speed": 1.0},
    {"steering_angle": 25.0, "speed": 1.0},
    {"steering_angle": 27.5, "speed": 1.0},
    {"steering_angle": 30.0, "speed": 1.0}
  ],
  "sensor": ["FRONT_FACING_CAMERA"],
  "neural_network": "DEEP_CONVOLUTIONAL_NETWORK_SHALLOW",
  "training_algorithm": "clipped_ppo",
  "action_space_type": "discrete",
  "version": "6"
}
class PARAMS:
    prev_speed = None
    prev_steering_angle = None 
    prev_steps = None
    prev_direction_diff = None
    prev_normalized_distance_from_route = None
    unpardonable_action = False
    intermediate_progress = [0] * 11

def reward_function(params):
    # Constants
    MAX_REWARD = 1e3
    TIME_PENALTY_FACTOR = 0.01  # Adjust this factor to control time penalty
    OFF_TRACK_PENALTY_MIN = 0.1
    OFF_TRACK_PENALTY_THRESHOLD = 0.5

    # Read input parameters
    heading = params['heading']
    distance_from_center = params['distance_from_center']
    steps = params['steps']
    steering_angle = params['steering_angle']
    speed = params['speed']
    progress = params.get('progress', 0)
    normalized_car_distance_from_route = params.get('normalized_car_distance_from_route', 0)
    vehicle_x = params['x']
    vehicle_y = params['y']
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    is_turn_upcoming = params.get('is_turn_upcoming', False)
    all_wheels_on_track = params['all_wheels_on_track']
    wheels_on_track = params.get('wheels_on_track', 4)
    track_width = params.get('track_width', 1.0)  # Example default value

    # Calculate the next waypoint
    next_point = waypoints[closest_waypoints[1]]

    # Adaptive Speed Reward (Suggestion 1)
    speed_reward = calculate_adaptive_speed_reward(speed)

    # Lateral Distance Reward Scaling (Suggestion 2)
    lateral_distance_reward = max(0, 1 - 2 * abs(distance_from_center) / track_width)

    # Off-Track Penalty
    off_track_penalty = max(OFF_TRACK_PENALTY_MIN, 1 - abs(normalized_car_distance_from_route))
    
    # Heading Reward
    heading_reward = calculate_heading_reward(heading, vehicle_x, vehicle_y, next_point)

    # Curvature-based Reward Adjustment (Suggestion 3)
    curvature_reward = calculate_curvature_reward(waypoints, closest_waypoints, speed)

    # Intermediate Progress Bonus Normalization (Suggestion 4)
    intermediate_progress_bonus = calculate_intermediate_progress_bonus(progress, steps, track_width)

    # Steering Angle Maintenance Bonus Scaling (Suggestion 5)
    steering_angle_bonus = calculate_steering_angle_bonus(steering_angle, speed)

    # Adaptive Time Penalty (Suggestion 6)
    adaptive_time_penalty = TIME_PENALTY_FACTOR * steps
    if distance_from_center > track_width / 4:
        adaptive_time_penalty *= 2  # Increase penalty if car is far from track center

    # Wheel-off-track Penalty Scaling (Suggestion 7)
    wheel_off_track_penalty = calculate_wheel_off_track_penalty(all_wheels_on_track, wheels_on_track)

    # Total reward calculation with weighted components
    total_reward = (
        0.3 * speed_reward +
        0.2 * lateral_distance_reward +
        0.2 * heading_reward +
        0.2 * curvature_reward +
        0.1 * intermediate_progress_bonus +
        steering_angle_bonus
    ) * off_track_penalty - adaptive_time_penalty - wheel_off_track_penalty

    # Reward Normalization (Suggestion 8)
    total_reward = normalize_reward(total_reward, progress, speed, track_width)

    # Reward Clipping (Suggestion 10)
    total_reward = min(max(total_reward, -MAX_REWARD), MAX_REWARD)

    return total_reward


def calculate_adaptive_speed_reward(speed):
    # Adaptive Speed Reward with a piecewise function (Suggestion 1)
    min_speed = 1.0
    optimal_speed_low = 2.0
    optimal_speed_high = 3.5
    max_speed = 4.0

    if speed < min_speed:
        return 0
    elif min_speed <= speed <= optimal_speed_low:
        return (speed - min_speed) / (optimal_speed_low - min_speed)
    elif optimal_speed_low < speed <= optimal_speed_high:
        return 1.0  # Full reward in the optimal range
    else:
        return max(0, (max_speed - speed) / (max_speed - optimal_speed_high))


def calculate_curvature_reward(waypoints, closest_waypoints, speed):
    # Refine curvature reward based on upcoming turns and speed (Suggestion 3)
    next_point = waypoints[closest_waypoints[1]]
    prev_point = waypoints[closest_waypoints[0]]
    
    dx = next_point[0] - prev_point[0]
    dy = next_point[1] - prev_point[1]
    curvature = math.atan2(dy, dx)

    if abs(curvature) < math.pi / 8:
        return 1.0  # Reward straight paths
    elif math.pi / 8 <= abs(curvature) < math.pi / 4:
        return 1.5 if speed < 2.5 else 1.0  # Reward low-speed navigation on curves
    else:
        return max(0, 1 - abs(curvature) / (math.pi / 4))  # Penalize sharp turns at high speeds


def calculate_intermediate_progress_bonus(progress, steps, track_width):
    # Normalize intermediate progress bonus by track complexity (Suggestion 4)
    progress_reward = 10 * progress / steps
    if steps <= 5:
        progress_reward = 1  # Ignore progress in the first 5 steps

    intermediate_progress_bonus = 0
    pi = int(progress // 10)
    if pi != 0 and PARAMS.intermediate_progress[pi] == 0:
        intermediate_progress_bonus = progress_reward ** (1 + 0.5 * pi)
        PARAMS.intermediate_progress[pi] = intermediate_progress_bonus

    return intermediate_progress_bonus


def calculate_steering_angle_bonus(steering_angle, speed):
    # Scale steering angle bonus based on speed (Suggestion 5)
    if PARAMS.prev_steering_angle is not None:
        STEERING_ANGLE_MAINTAIN_BONUS_MULTIPLIER = 2
        angle_diff = abs(steering_angle - PARAMS.prev_steering_angle)
        steering_angle_bonus = max(0, 1 - angle_diff / 10) * (STEERING_ANGLE_MAINTAIN_BONUS_MULTIPLIER * (speed / 3))
    else:
        steering_angle_bonus = 0

    PARAMS.prev_steering_angle = steering_angle
    return steering_angle_bonus


def calculate_wheel_off_track_penalty(all_wheels_on_track, wheels_on_track):
    # Scale penalty based on the number of wheels off the track (Suggestion 7)
    if all_wheels_on_track:
        return 0
    elif wheels_on_track == 3:
        return 0.1
    elif wheels_on_track == 2:
        return 0.25
    elif wheels_on_track == 1:
        return 0.5
    else:
        return 1.0  # Heavy penalty if all wheels are off the track


def normalize_reward(total_reward, progress, speed, track_width):
    # Normalize reward based on various factors (Suggestion 8)
    progress_factor = (progress + 1e-3)
    speed_factor = (speed + 1e-3)
    track_complexity_factor = max(0.5, track_width / 5)

    return total_reward / (progress_factor * speed_factor * track_complexity_factor)

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