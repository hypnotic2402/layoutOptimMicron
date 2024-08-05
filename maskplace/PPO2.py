import argparse
import pickle
from collections import namedtuple
import cv2

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import matplotlib.pyplot as plt

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import place_env
import torchvision
from place_db import PlaceDB
import time
from tqdm import tqdm
import random
from comp_res import comp_res
from torch.utils.tensorboard import SummaryWriter   
import json

import dgl
import dgl.function as fn

# CUDNN produces runtime error 
torch.backends.cudnn.enabled = False


# set device to cpu or cuda
device = torch.device('cuda')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    # device = torch.device('cpu')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")



from PIL import Image

import time
import imageio
import os

def create_video(image_folder, output_video_name, fps):
    images = []
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith('.png'):
            filepath = os.path.join(image_folder, filename)
            images.append(imageio.imread(filepath))
    imageio.mimsave(output_video_name, images, fps=fps)


# def train_agent(env, agent, benchmark, hpwl, cost, training_records, i_epoch, running_reward, score, writer, placed_num_macro, strftime, fwrite):
#     fwrite.close()
#     strftime_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
#     env.save_fig("./figures/{}-{}-{}-{}.png".format(benchmark, strftime_now, int(hpwl), int(cost)))
    
#     training_records.append(TrainingRecord(i_epoch, running_reward))
#     if i_epoch % 1 ==0:
#         print("Epoch {}, Moving average score is: {:.2f} ".format(i_epoch, running_reward))
#         fwrite.write("{},{},{:.2f},{}\n".format(i_epoch, score, running_reward, agent.training_step))
#         fwrite.flush()
#     writer.add_scalar('reward', running_reward, i_epoch)
#     if running_reward > -100:
#         print("Solved! Moving average score is now {}!".format(running_reward))
#         env.close()
#         agent.save_param()
#         return
#     if i_epoch % 100 == 0:
#         if placed_num_macro is None:
#             env.write_gl_file("./gl/{}{}.gl".format(strftime, int(score)))


# Parameters
parser = argparse.ArgumentParser(description='Solve the Pendulum-v0 with PPO')
parser.add_argument(
    '--gamma', type=float, default=0.95, metavar='G', help='discount factor (default: 0.9)')
parser.add_argument('--seed', type=int, default=42, metavar='N', help='random seed (default: 0)')
parser.add_argument('--disable_tqdm', type=int, default=1)
parser.add_argument('--lr', type=float, default=2.5e-3)
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    metavar='N',
    help='interval between training status logs (default: 10)')
parser.add_argument('--pnm', type=int, default=128)
parser.add_argument('--benchmark', type=str, default='adaptec1')
parser.add_argument('--soft_coefficient', type=float, default = 1)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--is_test', action='store_true', default=False)
parser.add_argument('--save_fig', action='store_true', default=False)
parser.add_argument('--enable_gcn', action='store_true', default=False)
args = parser.parse_args()
writer = SummaryWriter('./tb_log_with_GCN1')

benchmark = args.benchmark
placedb = PlaceDB(benchmark)
grid = 224
placed_num_macro = args.pnm
if args.pnm > placedb.node_cnt:
    placed_num_macro = placedb.node_cnt
    args.pnm = placed_num_macro
env = gym.make('place_env-v0', placedb = placedb, placed_num_macro = placed_num_macro, grid = grid).unwrapped

num_emb_state = 64 + 2 + 1
num_state = 1 + grid*grid*5 + 2

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

num_action = env.action_space.shape
seed_torch(args.seed)

Transition = namedtuple('Transition',['state', 'action', 'reward', 'a_log_prob', 'next_state', 'reward_intrinsic'])
TrainingRecord = namedtuple('TrainRecord',['episode', 'reward'])
print("seed = {}".format(args.seed))
print("lr = {}".format(args.lr))
print("placed_num_macro = {}".format(args.pnm))

# Encoder of Wire Mask
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 8, 1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 1),
        )
    def forward(self, x):
        return self.cnn(x)

# GCN
gcn_msg = fn.copy_u(u='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')
def make_edges():
    edges1 = []
    edges2 = []
    for net_name in placedb.net_info:
        st = set()
        for node_name in placedb.net_info[net_name]:
            for node2 in st:
                edges1.append(placedb.node_info[node_name]["id"])
                edges2.append(placedb.node_info[node2]["id"])
                edges1.append(placedb.node_info[node2]["id"])
                edges2.append(placedb.node_info[node_name]["id"])
    
    # store edges1 and edges2 in a file
    path1 = os.path.dirname(os.path.abspath(__file__)) + '/data/edges_1.dat'
    path2 = os.path.dirname(os.path.abspath(__file__)) + '/data/edges_2.dat'
    with open(path1, "w") as f:
        json.dump(edges1, f)
    with open(path2, "w") as f:
        json.dump(edges2, f)
    return edges1, edges2


def build_graph():
    x1, x2 = make_edges()

    g = dgl.DGLGraph()
    g.add_nodes(placedb.node_cnt)
    g.add_edges(x1, x2)
    # edges are directional in DGL; make them bi-directional
    g.add_edges(x2, x1)

    return g

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)



class CircuitTrainingModel(nn.Module):
    EPSILON = 1e-6

    def __init__(
            self, 
            num_gcn_layers=3, 
            edge_fc_layers=1,
            macro_features_dim=4, 
            gcn_node_dim=32, 
            max_macro_num=80,
            include_min_max_var=True, 
            is_augmented=False):
        
        super(CircuitTrainingModel, self).__init__()
        self.num_gcn_layers = num_gcn_layers
        self.gcn_node_dim = gcn_node_dim
        self.include_min_max_var = include_min_max_var
        self.is_augmented = is_augmented
        self.max_macro_num = max_macro_num
        self.macro_features_dim = macro_features_dim

        self.metadata_encoder = nn.Sequential(
            nn.Linear(self.gcn_node_dim, self.gcn_node_dim),
            nn.ReLU()
        )

        self.feature_encoder = nn.Sequential(
            nn.Linear(self.macro_features_dim, self.gcn_node_dim),
            nn.ReLU()
        )

        self.edge_fc_list = nn.ModuleList([
            self.create_edge_fc(edge_fc_layers) for _ in range(num_gcn_layers)
        ])

        self.attention_layer = nn.MultiheadAttention(embed_dim=self.gcn_node_dim, num_heads=1)
        self.attention_query_layer = nn.Linear(self.gcn_node_dim, self.gcn_node_dim)
        self.attention_key_layer = nn.Linear(self.gcn_node_dim, self.gcn_node_dim)
        self.attention_value_layer = nn.Linear(self.gcn_node_dim, self.gcn_node_dim)

        self.value_head = nn.Sequential(
            nn.Linear(self.gcn_node_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

        if self.is_augmented:
            self.augmented_embedding_layer = nn.Linear(self.gcn_node_dim, self.gcn_node_dim)

    def create_edge_fc(self, edge_fc_layers):
        layers = []
        layers.append(nn.Linear(2*self.gcn_node_dim+1, self.gcn_node_dim))
        layers.append(nn.ReLU())
        for _ in range(edge_fc_layers-1):
            layers.append(nn.Linear(self.gcn_node_dim, self.gcn_node_dim))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def gather_to_edges(self, h_nodes, sparse_adj_i, sparse_adj_j, sparse_adj_weight):
       h_edges_1 = h_nodes.gather(1, sparse_adj_i.unsqueeze(-1).expand(-1, -1, h_nodes.size(-1)))
        h_edges_2 = h_nodes.gather(1, sparse_adj_j.unsqueeze(-1).expand(-1, -1, h_nodes.size(-1)))
        sparse_adj_weight = sparse_adj_weight.unsqueeze(-1)
        h_edges_12 = torch.cat([h_edges_1, h_edges_2, sparse_adj_weight], dim=-1)
        h_edges_21 = torch.cat([h_edges_2, h_edges_1, sparse_adj_weight], dim=-1)
        mask = sparse_adj_weight.squeeze(-1) != 0.0
        h_edges_i_j = torch.where(mask.unsqueeze(-1), h_edges_12, torch.zeros_like(h_edges_12))
        h_edges_j_i = torch.where(mask.unsqueeze(-1), h_edges_21, torch.zeros_like(h_edges_21))

        return h_edges_i_j, h_edges_j_i

     def scatter_to_nodes(self, h_edges, sparse_adj_i, sparse_adj_j, num_nodes):
        batch_size, num_edges, feature_dim = h_edges.size()
        max_num_nodes = self.max_macro_num  
        
        h_nodes_1 = torch.zeros(batch_size, max_num_nodes, feature_dim, device=h_edges.device)
        count_1 = torch.zeros(batch_size, max_num_nodes, device=h_edges.device)
        
        h_nodes_2 = torch.zeros(batch_size, max_num_nodes, feature_dim, device=h_edges.device)
        count_2 = torch.zeros(batch_size, max_num_nodes, device=h_edges.device)

        for b in range(batch_size):
            h_nodes_1[b].index_add_(0, sparse_adj_i[b], h_edges[b])
            count_1[b].index_add_(0, sparse_adj_i[b], torch.ones_like(sparse_adj_i[b], dtype=torch.float, device=h_edges.device))

        for b in range(batch_size):
            h_nodes_2[b].index_add_(0, sparse_adj_j[b], h_edges[b])
            count_2[b].index_add_(0, sparse_adj_j[b], torch.ones_like(sparse_adj_j[b], dtype=torch.float, device=h_edges.device))

        h_nodes = (h_nodes_1 + h_nodes_2) / (count_1.unsqueeze(-1) + count_2.unsqueeze(-1) + self.EPSILON)

        return h_nodes


    def forward(self, inputs):
        sparse_adj_i = inputs['sparse_adj_i']
        sparse_adj_j = inputs['sparse_adj_j']
        sparse_adj_weight = inputs['sparse_adj_weight']
        node_features = inputs['node_features']
        h_nodes = self.feature_encoder(node_features)
        for i in range(self.num_gcn_layers):
            # print(h_nodes.shape)
            h_edges_i_j, h_edges_j_i = self.gather_to_edges(h_nodes, sparse_adj_i, sparse_adj_j, sparse_adj_weight)
            # print(h_edges_i_j.shape, h_edges_j_i.shape)
            h_edges = (self.edge_fc_list[i](h_edges_i_j) + self.edge_fc_list[i](h_edges_j_i)) / 2.0
            h_nodes_new = self.scatter_to_nodes(h_edges, sparse_adj_i, sparse_adj_j, inputs['num_nodes'])
            h_nodes = h_nodes_new + h_nodes
            # print(h_nodes.shape, h_nodes_new.shape)

        return h_nodes, h_edges


# decoder
class MyCNNCoarse(nn.Module):
    def __init__(self, res_net):
        super(MyCNNCoarse, self).__init__()
        self.cnn = res_net.to(device)
        self.cnn.fc = torch.nn.Linear(512, 16*7*7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding = 1), #14
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1, output_padding = 1), #28
            nn.ReLU(),
            nn.ConvTranspose2d(4, 2, 3, stride=2, padding=1, output_padding = 1), #56
            nn.ReLU(),
            nn.ConvTranspose2d(2, 1, 3, stride=2, padding=1, output_padding = 1), #112
            nn.ReLU(),
            nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1, output_padding = 1), #224
        )
    def forward(self, x):
        x = self.cnn(x).reshape(-1, 16, 7, 7)
        return self.deconv(x)

# Actor-Critic Network
class Actor(nn.Module):
    def __init__(self, cnn, gcn, cnn_coarse):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_emb_state, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, grid * grid)
        self.cnn = cnn
        self.cnn_coarse = cnn_coarse
        self.gcn = gcn
        self.softmax = nn.Softmax(dim=-1)
        if args.enable_gcn:
            # self.merge = nn.Conv2d(3, 1, 1)
            # self.gcn_layer1 = GCNLayer(6, 64)
            # self.gcn_layer2 = GCNLayer(64, 256)
            # self.gcn_layer3 = GCNLayer(256, grid * grid)
            # self.g = build_graph().to(device)
            pass
        else:
            self.merge = nn.Conv2d(2, 1, 1)

    def forward(self, x, graph_features = None, cnn_res = None, gcn_res = None, graph_node = None):
        if not cnn_res:
            cnn_input = x[:, 1+grid*grid*1: 1+grid*grid*5].reshape(-1, 4, grid, grid)
            mask = x[:, 1+grid*grid*2: 1+grid*grid*3].reshape(-1, grid, grid)
            mask = mask.flatten(start_dim=1, end_dim=2)
            cnn_res = self.cnn(cnn_input)
            coarse_input = torch.cat((x[:, 1: 1+grid*grid*2].reshape(-1, 2, grid, grid),
                                        x[:, 1+grid*grid*3: 1+grid*grid*4].reshape(-1, 1, grid, grid)
                                        ),dim= 1).reshape(-1, 3, grid, grid)
            cnn_coarse_res = self.cnn_coarse(coarse_input)
            # cnn_res = self.merge(torch.cat((cnn_res, cnn_coarse_res), dim=1))
            if args.enable_gcn:
                f1 = self.gcn_layer1(self.g, graph_features)
                f2 = self.gcn_layer2(self.g, f1)
                f3 = self.gcn_layer3(self.g, f2)
                gcn_res = f3
                f3 = f3[x[0, 0].long()].reshape(-1, 1, grid, grid)
                f3 = f3.repeat(cnn_res.size(0), 1, 1, 1)
                combined_features = torch.cat((cnn_res, cnn_coarse_res, f3), dim=1)
            else:
                combined_features = torch.cat((cnn_res, cnn_coarse_res), dim=1)
            cnn_res = self.merge(combined_features)

        net_img = x[:, 1+grid*grid: 1+grid*grid*2]
        net_img = net_img + x[:, 1+grid*grid*2: 1+grid*grid*3] * 10
        net_img_min = net_img.min() + args.soft_coefficient
        mask2 = net_img.le(net_img_min).logical_not().float()

        x = cnn_res
        x = x.reshape(-1, grid * grid)
        x = torch.where(mask + mask2 >=1.0, -1.0e10, x.double())
        x = self.softmax(x)

        return x, cnn_res, gcn_res


class Critic(nn.Module):
    def __init__(self, cnn, gcn, cnn_coarse, res_net):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.state_value = nn.Linear(64, 1)
        self.pos_emb = nn.Embedding(1400, 64)
        self.cnn = cnn
        self.gcn = gcn
    def forward(self, x, graph = None, cnn_res = None, gcn_res = None, graph_node = None):
        # print(x[:, 0]) # torch.Size([64, 250883])
        x1 = F.relu(self.fc1(self.pos_emb(x[:, 0].long())))
        x2 = F.relu(self.fc2(x1))
        value = self.state_value(x2)
        return value


class PPO():
    clip_param = 0.2 
    max_grad_norm = 0.5 
    ppo_epoch = 10 
    if placed_num_macro:
        buffer_capacity = 10 * (placed_num_macro)
    else:
        buffer_capacity = 5120
    batch_size = args.batch_size
    print("batch_size = {}".format(batch_size))

    def __init__(self):
        super(PPO, self).__init__()
        self.gcn = None
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.cnn = MyCNN().to(device)
        self.cnn_coarse = MyCNNCoarse(self.resnet).to(device)
        self.actor_net = Actor(cnn = self.cnn, gcn = self.gcn, cnn_coarse = self.cnn_coarse).float().to(device)
        self.critic_net = Critic(cnn = self.cnn, gcn = self.gcn,  cnn_coarse = None, res_net = self.resnet).float().to(device)
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), args.lr)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), args.lr)
        self.features = torch.zeros((placed_num_macro, 6), dtype=torch.float).to(device)

        actor_params = sum(p.numel() for p in self.actor_net.parameters() if p.requires_grad)
        critic_params = sum(p.numel() for p in self.critic_net.parameters() if p.requires_grad)
        print(f'Number of parameters in actor_net: {actor_params}')
        print(f'Number of parameters in critic_net: {critic_params}')

    def load_param(self, path):
        checkpoint = torch.load(path, map_location=torch.device(device))
        self.actor_net.load_state_dict(checkpoint['actor_net_dict'])
        self.critic_net.load_state_dict(checkpoint['critic_net_dict'])
    
    def select_action(self, state, info=None):
        if info:
            macro_iter = info["iter"]
            self.features[macro_iter][0] = info['state_size_x']
            self.features[macro_iter][1] = info['state_size_y']
            self.features[macro_iter][2] = info['pin_num']
            self.features[macro_iter][3] = info['state_size_x'] * info['state_size_y']

        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        with torch.no_grad():
            action_probs, _, _ = self.actor_net(state, graph_features=self.features)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.item(), action_log_prob.item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self, running_reward):
        strftime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        if not os.path.exists("save_models"):
            os.mkdir("save_models")
        torch.save({"actor_net_dict": self.actor_net.state_dict(),
                    "critic_net_dict": self.critic_net.state_dict()},
                    "./save_models/with_gcn_net_dict-{}-{}-".format(benchmark, placed_num_macro)+strftime+"{}".format(int(running_reward))+".pkl")

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter+=1
        return self.counter % self.buffer_capacity == 0

    def update(self):
        state = torch.tensor(np.array([t.state for t in self.buffer]), dtype=torch.float)
        action = torch.tensor(np.array([t.action for t in self.buffer]), dtype=torch.float).view(-1, 1).to(device)
        reward = torch.tensor(np.array([t.reward for t in self.buffer]), dtype=torch.float).view(-1, 1).to(device)
        old_action_log_prob = torch.tensor(np.array([t.a_log_prob for t in self.buffer]), dtype=torch.float).view(-1, 1).to(device)
        del self.buffer[:]
        target_list = []
        target = 0
        for i in range(reward.shape[0]-1, -1, -1):
            if state[i, 0] >= placed_num_macro - 1:
                target = 0
            r = reward[i, 0].item()
            target = r + args.gamma * target
            target_list.append(target)
        target_list.reverse()
        target_v_all = torch.tensor(np.array([t for t in target_list]), dtype=torch.float).view(-1, 1).to(device)
        for _ in tqdm(range(self.ppo_epoch)): # iteration ppo_epoch 
            for index in tqdm(BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, True),
                disable = args.disable_tqdm):
                self.training_step +=1
                
                action_probs, _, _ = self.actor_net(state[index].to(device), graph_features=self.features)
                dist = Categorical(action_probs)
                action_log_prob = dist.log_prob(action[index].squeeze())
                ratio = torch.exp(action_log_prob - old_action_log_prob[index].squeeze())
                target_v = target_v_all[index]                
                critic_net_output = self.critic_net(state[index].to(device))
                advantage = (target_v - critic_net_output).detach()

                L1 = ratio * advantage.squeeze() 
                L2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * advantage.squeeze() 
                action_loss = -torch.min(L1, L2).mean() # MAX->MIN desent

                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = F.smooth_l1_loss(self.critic_net(state[index].to(device)), target_v)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()

                writer.add_scalar('action_loss', action_loss, self.training_step)
                writer.add_scalar('value_loss', value_loss, self.training_step)


def save_placement(file_path, node_pos, ratio):
    fwrite = open(file_path, 'w')
    node_place = {}
    for node_name in node_pos:

        x, y,_ , _ = node_pos[node_name]
        x = round(x * ratio + ratio) 
        y = round(y * ratio + ratio)
        node_place[node_name] = (x, y)
    print("len node_place", len(node_place))
    for node_name in placedb.node_info:
        if node_name not in node_place:
            continue
        x, y = node_place[node_name]
        fwrite.write('{}\t{}\t{}\t:\tN /FIXED\n'.format(node_name, x, y))
    print(".pl has been saved to {}.".format(file_path))


def main():

    agent = PPO()
    strftime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
    
    training_records = []
    running_reward = -1000000
    

    log_file_name = "logs/log_"+ benchmark + "_" + strftime + "_seed_"+ str(args.seed) + "_pnm_" + str(args.pnm) + ".csv"
    if not os.path.exists("logs"):
        os.mkdir("logs")
    fwrite = open(log_file_name, "w")
    # load_model_path = None
    load_model_path = './save_models/without_gcn_net_dict-adaptec1-38-2024-07-23-11-22-02-101750.pkl'

    if load_model_path:
       agent.load_param(load_model_path)
    
    best_reward = running_reward
    if args.is_test:
        torch.inference_mode()
    images = []
    wiremask_images = []
    for i_epoch in range(600):
        score = 0
        raw_score = 0
        start = time.time()
        state, info = env.reset()
        done = False
        step = 0
        while done is False:
            state_tmp = state.copy()
            action, action_log_prob = agent.select_action(state, info)
            agent.features[int(state[0])][4] = action // grid
            agent.features[int(state[0])][5] = action % grid

            next_state, reward, done, info = env.step(action)
            if args.is_test:
                image_path = "./figures/step_{}_{}.png".format(i_epoch, step)
                env.save_fig(image_path)  # Save image after each step
                wiremask_img_path = "./figures/wiremask_{}.png".format(step)
                wiremask_img = env.get_net_img()

                plt.imshow(wiremask_img, cmap='inferno', interpolation='nearest')

                # fig = plt.gcf()
                # fig.canvas.draw()
                # heatmap_image_np = np.array(fig.canvas.renderer._renderer)

                # Save the heatmap as an image
                heatmap_img_path = "./figures/heatmap_{}.png".format(step)
                plt.savefig(heatmap_img_path)

                image = Image.open(image_path)
                image_np = np.array(image)
                img_heatmap = Image.open(heatmap_img_path)
                img_heatmap_np = np.array(img_heatmap)
                images.append(image_np)
                wiremask_images.append(img_heatmap_np)

            assert next_state.shape == (num_state, )
            reward_intrinsic = 0
            if not args.is_test:
                trans = Transition(state_tmp, action, reward / 200.0, action_log_prob, next_state, reward_intrinsic)
            if not args.is_test and agent.store_transition(trans):                
                assert done == True
                agent.update()

            if args.is_test:
                env.save_fig("./figures/inter_{}.png".format(i_epoch))
                        
            score += reward
            raw_score += info["raw_reward"]
            state = next_state
            step += 1  # Increment step count
        end = time.time()
        # Convert the list of images to a numpy array
        if args.is_test:
            images_np = np.array(images)
            wiremask_images_np = np.array(wiremask_images)  # Convert the list of images to a numpy array
            # Create a video from the numpy array of images
            # imageio.mimsave('./figures/wiremask_{}.mp4'.format(i_epoch), wiremask_images_np, fps=2)
            print("Creating video...")
            imageio.mimsave('./figures/floorplan_evolution.mp4', images_np, fps=2)
            imageio.mimsave('./figures/wiremask_evolution.mp4', wiremask_images_np, fps=2)
        
        if i_epoch == 0:
            running_reward = score
        running_reward = running_reward * 0.9 + score * 0.1
        print("score = {}, raw_score = {}".format(score, raw_score))

        if running_reward > best_reward * 0.975:
            best_reward = running_reward
            if i_epoch >= 10:
                agent.save_param(running_reward)
                if args.save_fig:
                    strftime_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                    if not os.path.exists("figures"):
                        os.mkdir("figures")
                    env.save_fig("./figures/{}{}.png".format(strftime_now,int(raw_score)))
                    print("save_figure: figures/{}{}.png".format(strftime_now,int(raw_score)))
                try:
                    print("start try")
                    # cost is the routing estimation based on the MST algorithm
                    hpwl, cost = comp_res(placedb, env.node_pos, env.ratio)
                    print("hpwl = {:.2f}\tcost = {:.2f}".format(hpwl, cost))
                except:
                    assert False
        
        if i_epoch % 10 == 0:
            if args.save_fig:
                    strftime_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                    if not os.path.exists("figures"):
                        os.mkdir("figures")
                    env.save_fig("./figures/inter_10.png")
                    print("save_figure: figures/inter_10.png".format(strftime_now,int(raw_score)))

        if args.is_test:
            print("save node_pos")
            hpwl, cost = comp_res(placedb, env.node_pos, env.ratio)
            print("hpwl = {:.2f}\tcost = {:.2f}".format(hpwl, cost))
            print("time = {}s".format(end-start))
            pl_file_path = "{}-{}-{}.pl".format(benchmark, int(hpwl), time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) ) 
            save_placement(pl_file_path, env.node_pos, env.ratio)
            strftime_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            pl_path = 'gg_place_new/{}-{}-{}-{}.pl'.format(benchmark, strftime_now, int(hpwl), int(cost))
            fwrite_pl = open(pl_path, 'w')
            for node_name in env.node_pos:
                if node_name == "V":
                    continue
                x, y, size_x, size_y = env.node_pos[node_name]
                x = x * env.ratio + placedb.node_info[node_name]['x'] /2.0
                y = y * env.ratio + placedb.node_info[node_name]['y'] /2.0
                fwrite_pl.write("{}\t{:.4f}\t{:.4f}\n".format(node_name, x, y))
            fwrite_pl.close()
            strftime_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            env.save_fig("./figures/{}-{}-{}-{}.png".format(benchmark, strftime_now, int(hpwl), int(cost)))
            break

        training_records.append(TrainingRecord(i_epoch, running_reward))
        if i_epoch % 1 ==0:
            print("Epoch {}, Moving average score is: {:.2f} ".format(i_epoch, running_reward))
            fwrite.write("{},{},{:.2f},{}\n".format(i_epoch, score, running_reward, agent.training_step))
            fwrite.flush()
        writer.add_scalar('reward', running_reward, i_epoch)
        if running_reward > -100:
            print("Solved! Moving average score is now {}!".format(running_reward))
            env.close()
            agent.save_param()
            break
        if i_epoch % 100 == 0:
            if placed_num_macro is None:
                env.write_gl_file("./gl/{}{}.gl".format(strftime, int(score)))



if __name__ == '__main__':
    main()
