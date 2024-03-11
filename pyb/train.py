import pybullet as p
from numpy import random
import pybullet_data
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from multiprocessing import Process, Queue, Event, Value, Lock,freeze_support
import numpy as np
import os
from collections import namedtuple, deque
import itertools




p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# First, let's make sure we start with a fresh new simulation.
# Otherwise, we can keep adding objects by running this cell over again.
p.resetSimulation()

p.setPhysicsEngineParameter(fixedTimeStep=1/100)
# Set the gravity to Earth's gravity.
p.setGravity(0, 0, -9.807)

# Define the maximum velocity limit
max_velocity_limit =2.11604775654# Adjust as needed

batch_size=32



Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward','priority'))

class ReplayMemory(object):
    def __init__(self, capacity,batch_size):
        self.memory = deque([], maxlen=capacity)
        self.sequence_length = 128  # Define the length of each sequence
        self.batch_size=batch_size
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        sequences = []
        for _ in range(batch_size):
            while True:
                start = random.randint(0, len(self.memory) - self.sequence_length)
                sequence = list(itertools.islice(self.memory, start, start + self.sequence_length))
                # Check if sequence crosses episode boundary
                if not any(t.next_state is None for t in sequence):
                    sequences.append(sequence)
                    break
        print(len(sequences),"sequencec samples",'of lenth',len(sequences[0]))
        self.cleanup_low_reward_sequences()
        return sequences

    def calculate_sequence_reward(self, sequence):
        return sum(transition.reward for transition in sequence)

    def cleanup_low_reward_sequences(self):
        if len(self.memory) <= self.batch_size:
            return
    
        all_sequences = [list(itertools.islice(self.memory, start, start + self.sequence_length))
                         for start in range(len(self.memory) - self.sequence_length + 1)]
    
        # Compute rewards for each sequence
        sequence_rewards = [self.calculate_sequence_reward(seq) for seq in all_sequences]
    
        # Determine the number of sequences to remove (25% of sequences beyond batch size)
        num_sequences_to_remove = (len(all_sequences) - self.batch_size) // 4
    
        # Get indices of the lowest-reward sequences
        lowest_reward_indices = sorted(range(len(sequence_rewards)), key=lambda i: sequence_rewards[i])[:num_sequences_to_remove]
    
        # Remove these sequences by removing their starting transitions
        for idx in sorted(lowest_reward_indices, reverse=True):
            del self.memory[idx]

    
    def __len__(self):
        return len(self.memory)



if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("GPU is not available. Using CPU instead.")





class CustomSquash(nn.Module):
    def forward(self, x):
        # Squash the output to be in the range [-0.5, 0.5]
        return torch.tanh(x)

class QuadrupedTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, nhead=4, dropout=0.5):
        super(QuadrupedTransformer, self).__init__()
        
        self.embedding = nn.Linear(input_size, hidden_size)
        transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.custom_activation = CustomSquash()

    
    def forward(self, x):
        # x should be of shape (batch_size, sequence_length, input_size)
        x = self.embedding(x) # Transform to hidden size

        # The Transformer expects input of shape (sequence_length, batch_size, hidden_size)
        x = x.permute(1, 0, 2)
        x = torch.tanh(x)

        # Forward propagate through the Transformer
        out = self.transformer_encoder(x)

        # Decode the hidden state of the last time step
        out = self.fc(out[-1, :, :])
        out = self.custom_activation(out)
        return out   
    



p.resetSimulation()

# Load our simulation floor plane at the origin (0, 0, 0).




    
def calculate_reward(r2d2,ground,position, orientation, joint_states, linear_vel, angular_vel,target_joint_positions, initial_position, initial_orientation,foot_joint_indices = [3, 5, 9, 11,2,4,8,10]):
    # Constants
    HEIGHT_TARGET = 2.2  # Target height for the robot
    HEIGHT_WEIGHT = 100 # Weight for height reward
    ORIENTATION_WEIGHT =166.405104687  # Weight for orientation penalty
    POSITION_WEIGHT = 0 # Weight for position penalty
    MOVEMENT_WEIGHT =100.45  # Weight for movement penalty
    CONTACT_POINTS_PENALTY_WEIGHT = 85.16548  # Weight for contact points reward/penalty
    CONTACT_POINTS_REWARD_WEIGHT = 168.553
    LEG_POSITION_PENALTY_WEIGHT = 164.35130  # Adjust as needed
    TORQUE_PENALTY_WEIGHT = 4.450

    x, y, z = position
    initial_x, initial_y, _ = initial_position

    # Height Reward
    height_reward = -HEIGHT_WEIGHT * abs(z - HEIGHT_TARGET)

    # Orientation Penalty
    orientation_error = np.linalg.norm(np.array(orientation) - np.array(initial_orientation))
    orientation_penalty = -ORIENTATION_WEIGHT * orientation_error

    # Position Penalty
    position_penalty = -POSITION_WEIGHT * (abs(x - initial_x) + abs(y - initial_y))

    # Movement Penalty
    joint_velocities = [state[1] for state in joint_states]
    movement_penalty = -MOVEMENT_WEIGHT * (np.linalg.norm(joint_velocities) + np.linalg.norm(linear_vel) + np.linalg.norm(angular_vel))



    contact_reward, contact_penalty= calculate_contact_reward_penalty(r2d2,ground,joint_states)

    contact_reward = CONTACT_POINTS_REWARD_WEIGHT*contact_reward
    contact_penalty = -CONTACT_POINTS_PENALTY_WEIGHT*contact_penalty
    
    joint_torques = [state[3] for state in joint_states]  # Extract joint torques
    torque_penalty = -TORQUE_PENALTY_WEIGHT * sum(abs(torque) for torque in joint_torques)


    current_joint_positions = [joint_states[idx][0] for idx in foot_joint_indices]
    leg_dev = [abs(i-j) for i in current_joint_positions for j in target_joint_positions ]
    leg_position_penalty = -LEG_POSITION_PENALTY_WEIGHT*sum(leg_dev)  # Adjust as neededcurrent_joint_positions

    # Total Reward
    total_reward = height_reward + orientation_penalty + position_penalty + movement_penalty + torque_penalty + leg_position_penalty +contact_reward + contact_penalty


    

    return total_reward


def calculate_contact_reward_penalty(robot,ground,joint_states,leg_joint_indices=[3,9,5,11]):
    contact_reward = 0
    contact_penalty = 0

    for joint_index in range(len(joint_states)):
        # Check for contact with the ground
        contacts = p.getContactPoints(bodyA=robot, bodyB=ground, linkIndexA=joint_index)

        if contacts:
            if joint_index in leg_joint_indices:
                # Reward for leg joints making contact
                contact_reward += 1
            else:
                # Penalty for other joints making contact
                contact_penalty += 1

    return contact_reward, contact_penalty


    



import logging

def setup_logging():
    logging.basicConfig(filename='trakkkkining.log', level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(processName)s: %(message)s')



def train_robot(process_id, best_model_event,shared_performance,model_lock):

    setup_logging()
    logging.debug(f"Starting process {process_id}")
    ground = p.loadURDF('plane.urdf')

    
    replay_memory = ReplayMemory(capacity=4096,batch_size =batch_size)  # Adjust the capacity as needed


    
    # Initialize PyBullet, model, and other necessary components

    p.resetSimulation()
    
    # Load an R2D2 droid at the position at 0.5 meters height in the z-axis.
    r2d2 = p.loadURDF('1/bittle.urdf',  [0.4*process_id, 0, 2.6],globalScaling=3.5, flags=p.URDF_USE_SELF_COLLISION)

        
    for i in range(4000):
        p.stepSimulation()


    
    # Capture initial position
    initial_position, _ = p.getBasePositionAndOrientation(r2d2)
    initial_x, initial_y, _ = initial_position



    # Observations
    position, orientation = p.getBasePositionAndOrientation(r2d2)
    x, y, z = position
    roll, pitch, yaw = p.getEulerFromQuaternion(orientation)
    joint_states = p.getJointStates(r2d2, range(p.getNumJoints(r2d2)))
    joint_positions = [state[0] for state in joint_states]  # Joint positions
    joint_velocities = [0 for state in joint_states]  # Joint velocities
    contact_points = len(p.getContactPoints(bodyA=ground, bodyB=r2d2))
    
    joint_torques = [state[3] for state in joint_states]  # Joint torques
    linear_vel, angular_vel = p.getBaseVelocity(r2d2)
    
    # Combine all observations
    observations = [x, y, z, roll, pitch, yaw] + joint_positions + joint_velocities + [contact_points] +list(linear_vel)+list(angular_vel)+list(joint_torques)
    
    
    input_size = len(observations)  # This should be the length of your observation vector
    output_size = p.getNumJoints(r2d2) 
    
    

    
    # Example joint_states structure: [(position, velocity, reaction_forces, applied_effort), ...]
    foot_joint_indices = [3, 5, 9, 11,2,4,8,10]
    
    target_joint_positions = [joint_states[idx][0] for idx in foot_joint_indices]
    target_joint_positions



    initial_orientation = orientation
    
    
    # Run the simulation for a fixed amount of steps.
    observations = []
    
    
    # Initialize training variables
    train = True  # Set this to True or False
    training_data = []  # To store (observation, action, reward)


    model = QuadrupedTransformer(input_size,4, output_size)   
    if os.path.exists('shared_model.pth'):
        print("GETTING BEST MODEL FROM QUE for robot ",process_id)
        model.load_state_dict(torch.load('shared_model.pth'))
        best_model_event.clear()

        

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Example optimizer

    t=0
    tt=256
    train_alternator = -1 
    #start sim
    episode=1
    trains = 0
    while trains< 40:
        t+=1
        if t==tt:
            p.resetBasePositionAndOrientation(r2d2, initial_position, initial_orientation)
    
            for joint in  range(p.getNumJoints(r2d2)):
                p.resetJointState(r2d2, joint, targetValue=0, targetVelocity=0)
            t=0
            episode+=1
            # print(episode,":episode finised")
            continue
    
    
    
        ##OBSERVATIONS
        
        # Observations
        position, orientation = p.getBasePositionAndOrientation(r2d2)
        x, y, z = position
        roll, pitch, yaw = p.getEulerFromQuaternion(orientation)
        joint_states = p.getJointStates(r2d2, range(p.getNumJoints(r2d2)))
        joint_positions = [state[0] for state in joint_states]  # Joint positions
        joint_velocities = [state[1] for state in joint_states]  # Joint velocities
        contact_points = len(p.getContactPoints(bodyA=ground, bodyB=r2d2))
        linear_vel, angular_vel = p.getBaseVelocity(r2d2)
    
        joint_torques = [state[3] for state in joint_states]  # Joint torques
        
        # Combine all observations
        observations = [x, y, z, roll, pitch, yaw] + joint_positions + joint_velocities + [contact_points] + list(linear_vel) +list(angular_vel) +list(joint_torques)
        # print(contact_points)
        # # Convert observations to a PyTorch tensor
        # observations_tensor = torch.tensor(observations, dtype=torch.float32)
    
    
        observations_tensor = torch.tensor(observations, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        observations_tensor = observations_tensor.to(device)
        actions = model(observations_tensor).detach().cpu().numpy()[0]
    
    
        

        for i in range(p.getNumJoints(r2d2)):

           
            p.setJointMotorControl2(r2d2,jointIndex=i, controlMode=p.VELOCITY_CONTROL, targetVelocity=max_velocity_limit*actions[i])
    
    
    
        
        # next Observations
        position, orientation = p.getBasePositionAndOrientation(r2d2)
        x, y, z = position
        roll, pitch, yaw = p.getEulerFromQuaternion(orientation)
        joint_states = p.getJointStates(r2d2, range(p.getNumJoints(r2d2)))
        joint_positions = [state[0] for state in joint_states]  # Joint positions
        joint_velocities = [state[1] for state in joint_states]  # Joint velocities
        contact_points = len(p.getContactPoints(bodyA=ground, bodyB=r2d2))
        linear_vel, angular_vel = p.getBaseVelocity(r2d2)
        joint_torques = [state[3] for state in joint_states]  # Joint torques
        
        # Combine all observations
        next_observations = [x, y, z, roll, pitch, yaw] + joint_positions + joint_velocities + [contact_points] + list(linear_vel) +list(angular_vel) +list(joint_torques)
    
        # # Convert observations to a PyTorch tensor
        # next_observations_tensor = torch.tensor(next_observations, dtype=torch.float32)
        
        
        reward = calculate_reward(r2d2,ground,position, orientation, joint_states, linear_vel, angular_vel,target_joint_positions, initial_position, initial_orientation,)
    
        
        # Store data for training
        if train:
            # Convert observations and next_observations to PyTorch tensors
            observations_tensor = torch.tensor(observations, dtype=torch.float32).unsqueeze(0).to(device)
            next_observations_tensor = torch.tensor(next_observations, dtype=torch.float32).unsqueeze(0).to(device)
            actions_tensor = torch.tensor(actions, dtype=torch.float32).unsqueeze(0).to(device)
            reward_tensor = torch.tensor([reward], dtype=torch.float32).to(device)
        
            # Store the transition in replay memory
        
            # Calculate initial priority for the new experience
            # Using absolute reward as a simple priority measure
            initial_priority = abs(reward)
    
            # Store the transition with priority in replay memory
            replay_memory.push(observations_tensor, actions_tensor, next_observations_tensor, reward_tensor, initial_priority)
    
    

        if train and len(replay_memory) >= batch_size and episode%25==0 :
            episode+=1
            trains+=1
            train_alternator=train_alternator*-1
            print('training',train_alternator)
            gamma = 0.99
            sequences = replay_memory.sample(batch_size)

            if best_model_event.is_set():
                print("GETTING BEST MODEL FROM QUE for robot ",process_id)
                if os.path.exists('shared_model.pth'):
                    model.load_state_dict(torch.load('shared_model.pth'))
                    print('---------GOT BEST MODEL--------')
                    best_model_event.clear()
  
            for sequence in sequences:
    
                
                batch = Transition(*zip(*sequence))
            
                # Concatenate the batch elements into separate tensors
                state_batch = torch.cat(batch.state).unsqueeze(1)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)
                next_state_batch = torch.cat(batch.next_state).unsqueeze(1)
            
                # Predict current Q-values
                # print(state_batch.shape,reward_batch.shape)
                current_q_values = model(state_batch)
        
        
                
            
                # Predict next Q-values
                next_q_values = model(next_state_batch)
            
                # Average next Q-values across joints
                average_next_q_values = torch.mean(next_q_values, dim=1)
            
                # Calculate TD target
                td_target = reward_batch + gamma * average_next_q_values
            
                # Compute loss
                # print(current_q_values)
                loss = F.mse_loss(current_q_values, td_target.unsqueeze(1))
        
                
                # Perform optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # If this is the best model so far, share it with other processes
                # Evaluate the model
                performance = loss.cpu().detach().numpy()
                print('###loss',loss.cpu().detach().numpy())
                with model_lock:
                    print( shared_performance.value,"trained to ",performance)
                    if performance < shared_performance.value:
                        shared_performance.value = performance
                        torch.save(model.state_dict(), 'shared_model.pth')
                        best_model_event.set()
                        print('********-----PUT BEST MODEL IN QUE-----*******')
                        print(f'{performance} best performance')
        p.stepSimulation()
    
            








#     # Shared components
# Best_model = []

if __name__ == '__main__':
    freeze_support()  # For Windows support

    

    best_model_event = Event()
    shared_performance = Value('d', float('inf'))  # Shared variable for best performance
    model_lock = Lock()  # Lock for updating shared performance

    # Number of robots
    num_robots = 8

    processes = []
    for i in range(num_robots):
        print('starting robot',i)
        q = Process(target=train_robot, args=(i, best_model_event,shared_performance,model_lock))
        q.start()
        processes.append(q)

    for q in processes:
        print(f"Process {q.name} alive: {q.is_alive()}")
        q.join()


    #     # After all processes have completed
    # if Best_model!=[]:
    #     best_model = Best_model[0]  # Retrieve the best model from the queue
    #     torch.save(best_model.state_dict(), 'best_model.pth')  # Save the model
    #     print("Saved the best model as 'best_model.pth'")
    # else:
    #     print("No model found in the queue.")