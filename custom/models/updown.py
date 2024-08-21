from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_vector(v, dim, eps=1e-6):
    return v / (torch.linalg.norm(v, ord=2, dim=dim, keepdim=True) + eps)

def construct_3d_basis_from_1_vectors(v1):
    g = torch.ones_like(v1)
    g[..., 2] = 0
    v1 = g * v1
    u1 = normalize_vector(v1, dim=-1)

    u2 = torch.zeros_like(u1)
    u2[..., 0] = -u1[..., 1]
    u2[..., 1] = u1[..., 0]

    u3 = torch.zeros_like(u1)
    u3[..., 2] = 1

    mat = torch.stack([u1, u2, u3], dim=-1)  # (N, L, 3, 3)
    return mat

# def construct_3d_basis_from_1_vectors(v1):
#     g = torch.ones_like(v1)
#     g[..., 2] = 0
#     v1 = g * v1
#     # print("g",g)
#     u1 = normalize_vector(v1, dim=-1)
#     # print("u1",u1)

#     u2 = torch.zeros_like(u1)
#     u2[..., 0] = u1[..., 1]
#     u2[..., 1] = -u1[..., 0]
#     # e2 = normalize_vector(u2, dim=-1)
#     # print("u2",u2)

#     u3 = torch.cross(u1, u2, dim=-1)    # (N, L, 3)
#     u3 = torch.zeros_like(u1)
#     u3[..., 2] = 1

#     mat = torch.cat([
#         u1.unsqueeze(-1), u2.unsqueeze(-1), u3.unsqueeze(-1)
#     ], dim=-1)  # (N, L, 3, 3_index)
#     return mat


class BaseMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation, residual=False, last_act=False, flat=False):
        super(BaseMLP, self).__init__()
        self.residual = residual
        if flat:
            activation = nn.Tanh()
            hidden_dim = 4 * hidden_dim
        if residual:
            assert output_dim == input_dim
        if last_act:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim, output_dim),
                activation
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim, output_dim)
            )

    def forward(self, x):
        return self.mlp(x) if not self.residual else self.mlp(x) + x


class ActorUp(nn.Module):
    """a bottom-up module used in bothway message passing that only passes message to its parent"""

    def __init__(self, Z_dim, h_dim, msg_dim, max_children):
        super(ActorUp, self).__init__()
        self.msg_dim = msg_dim
        self.fc_Z1 = nn.Linear(Z_dim, msg_dim, bias=False)
        self.fc_h1 = nn.Linear(h_dim, msg_dim)
        self.fc_g1 = nn.Sequential(
                nn.Linear(msg_dim + msg_dim * max_children, msg_dim),
                nn.ReLU(),
                nn.Linear(msg_dim, 1),
            )
        self.fc_Z2 = nn.Linear(msg_dim + msg_dim * max_children+1, msg_dim, bias=False)
        self.mlp = BaseMLP(input_dim=msg_dim*msg_dim + msg_dim + msg_dim * max_children, hidden_dim=msg_dim*msg_dim, output_dim=msg_dim*msg_dim+msg_dim, activation=nn.ReLU(), last_act=False)

    def forward(self, Z0, h0, m_Z0, m_h0):

        m_Z = torch.cat(m_Z0, dim=-1)
        m_h = torch.cat(m_h0, dim=-1)

        Z = self.fc_Z1(Z0)

        h = self.fc_h1(h0)

        hm = torch.cat([h, m_h], dim=-1)
        hm = F.relu(hm)

        g = torch.zeros_like(Z0)[..., 0]
        g[..., 2] = 1
        g = g*self.fc_g1(hm)
        Zm = torch.cat([Z, m_Z, g.unsqueeze(-1)], dim=-1)
        Zm = self.fc_Z2(Zm)
        

        M = torch.einsum('bij,bjk->bik', Zm.transpose(-1, -2), Zm)
        M = M.reshape(M.shape[0], -1)  # [bz, 32*32]
        F_norm = torch.norm(M,dim=(-1), keepdim=True)+1.0
        M = torch.cat([M, hm], dim=-1)

        M = self.mlp(M)
        M = M / F_norm
        M_Z, M_h = M[..., :Zm.shape[-1]*self.msg_dim], M[..., Zm.shape[-1]*self.msg_dim:]
        M_Z = torch.einsum('bij,bjk->bik', Zm, M_Z.reshape(M.shape[0], Zm.shape[-1],-1))  

        return M_Z, M_h



class ActorDownAction(nn.Module):
    """a top-down module used in bothway message passing that passes messages to children and outputs action"""

    # input dim is state dim if only using top down message passing
    # if using bottom up and then top down, it is the node's outgoing message dim
    def __init__(self, msg_dim, max_children):
        super(ActorDownAction, self).__init__()
        self.msg_dim = msg_dim
        self.max_children = max_children
        self.fc_g1 = nn.Sequential(
            nn.Linear(msg_dim+msg_dim, msg_dim),
            nn.ReLU(),
            nn.Linear(msg_dim, 1),
        )
        self.fc_Z2 = nn.Linear(msg_dim + msg_dim + 1, msg_dim, bias=False)
        self.frame_base = BaseMLP(input_dim=msg_dim*msg_dim+2*msg_dim, hidden_dim=msg_dim*msg_dim, output_dim=msg_dim * 1, activation=nn.ReLU(), last_act=False)

        self.msg_base = BaseMLP(input_dim=msg_dim*msg_dim+2*msg_dim, hidden_dim=msg_dim*msg_dim * max_children, output_dim=msg_dim*msg_dim * max_children + msg_dim * max_children, activation=nn.ReLU(), last_act=False)

    def forward(self, Z0, h0, m_Z0, m_h0):
        hm = torch.cat([h0, m_h0], dim=-1)
        hm = F.relu(hm)

        g = torch.zeros_like(Z0)[..., 0]
        g[..., 2] = 1
        g = g*self.fc_g1(hm)
        Zm = torch.cat([Z0, m_Z0, g.unsqueeze(-1)], dim=-1)
        Zm = self.fc_Z2(Zm)
        M = torch.einsum('bij,bjk->bik', Zm.transpose(-1, -2), Zm)
        M = M.reshape(M.shape[0], -1)  # [bz, 32*32]
        F_norm = torch.norm(M,dim=(-1), keepdim=True)+1.0
        M = torch.cat([M, hm], dim=-1)

        frame = self.frame_base(M)
        frame = frame / F_norm
        frame = torch.einsum('bij,bjk->bik', Zm, frame.reshape(Zm.shape[0], Zm.shape[-1],-1)) 

        msg_down = self.msg_base(M)
        msg_down = msg_down / F_norm
        M_Z, M_h = msg_down[..., :Zm.shape[-1]*self.msg_dim*self.max_children], msg_down[..., Zm.shape[-1]*self.msg_dim*self.max_children:]
        M_Z = torch.einsum('bij,bjk->bik', Zm, M_Z.reshape(Zm.shape[0], Zm.shape[-1],-1)) 
        return frame, M_Z, M_h



class FrameGNN(nn.Module):
    """a weight-sharing dynamic graph policy that changes its structure based on different morphologies and passes messages between nodes"""

    def __init__(
        self,
        state_dim,
        attr_fixed_dim,
        attr_design_dim,
        msg_dim,
        # batch_size,
        # max_action,
        max_children,
    ):
        super(FrameGNN, self).__init__()
        self.num_limbs = 1
        self.msg_down_Z = [None] * self.num_limbs
        self.msg_down_h = [None] * self.num_limbs
        self.msg_up_Z = [None] * self.num_limbs
        self.msg_up_h = [None] * self.num_limbs
        self.Z_h = [None] * self.num_limbs
        self.frame = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
        # self.max_action = max_action
        self.msg_dim = msg_dim
        # self.batch_size = batch_size
        self.max_children = max_children
        self.state_dim = state_dim
        self.attr_fixed_dim = attr_fixed_dim
        self.attr_design_dim = attr_design_dim
        # self.action_dim = action_dim
        self.z_num = 4

        # assert self.action_dim == 1
        self.sNet = nn.ModuleList(
            [ActorUp(self.z_num, self.attr_fixed_dim+self.attr_design_dim+self.state_dim-self.z_num*3, msg_dim, max_children)] * self.num_limbs
        )

        self.actor = nn.ModuleList(
            [ActorDownAction(msg_dim, max_children)] * self.num_limbs
        )



    def forward(self, state):
        self.clear_buffer()
        self.input_state = state
        if len(state.shape) == 2:
            self.input_state = self.input_state.unsqueeze(0)
        self.batch_size = self.input_state.shape[0]
        if self.batch_size > 1:
            print("Batch size > 1 is not supported")
        
        # for i in range(self.num_limbs):
        #     self.input_state[i] = state[
        #         :, i * (self.attr_fixed_dim+self.attr_design_dim+self.state_dim) : (i + 1) * (self.attr_fixed_dim+self.attr_design_dim+self.state_dim)
        #     ]

        for i in range(self.num_limbs):
            self.bottom_up_transmission(i)

        for i in range(self.num_limbs):
            self.top_down_transmission(i)

        # for i in range(self.num_limbs):
        #     Z = self.input_state[i][..., :self.z_num*3]
        #     Z = Z.contiguous().view(self.batch_size, -1,3).transpose(-2, -1)
        #     axis = Z[...,5:8]

        #     # rad = math.pi/4
        #     # theta = math.pi*rad
        #     # O = torch.tensor([[math.cos(theta), -math.sin(theta), 0],
        #     #             [math.sin(theta), math.cos(theta), 0],
        #     #             [0, 0, 1]]).unsqueeze(0)
        #     # O = O.repeat(axis.shape[0],1,1)
        #     # axis = torch.einsum('bij,bjk->bik', O, axis)


        #     # rad = math.pi/4
        #     # theta = -math.pi*rad
        #     # O = torch.tensor([[math.cos(theta), -math.sin(theta), 0],
        #     #             [math.sin(theta), math.cos(theta), 0],
        #     #             [0, 0, 1]]).unsqueeze(0)
        #     # O = O.repeat(axis.shape[0],1,1)
        #     # self.action[i] = torch.einsum('bij,bjk->bik', O, self.action[i])
        #     # print("self.action[i]",self.action[i])


        #     # print(self.action[i].shape, axis.shape)
        #     # print("self.action[i]",self.action[i])
        #     output0 = torch.einsum('bij,bjk->bik', axis[...,0:1].transpose(-1, -2), self.action[i])
        #     output1 = torch.einsum('bij,bjk->bik', axis[...,1:2].transpose(-1, -2), self.action[i])
        #     output2 = torch.einsum('bij,bjk->bik', axis[...,2:3].transpose(-1, -2), self.action[i])


        #     self.action[i] = torch.cat([output0,output1,output2],-1)
        #     self.action[i] = self.action[i].contiguous().view(self.batch_size,-1)
            # print(i,self.action[i].shape,self.action[i])

        frame = torch.cat(self.frame,-1)
        # rad = math.pi/4
        # theta = -math.pi*rad
        # O = torch.tensor([[math.cos(theta), -math.sin(theta), 0],
        #             [math.sin(theta), math.cos(theta), 0],
        #             [0, 0, 1]]).unsqueeze(0)
        # O = O.repeat(frame.shape[0],1,1)
        # frame = torch.einsum('bij,bjk->bik', O, frame)

        # print("self.frame[0]",self.frame[0].shape)
        frame = frame.mean(dim=-1)
        # print("frame",frame.shape)
        frame = construct_3d_basis_from_1_vectors(frame)
        # print("frame",frame)
        
        Z_all = []
        for i in range(self.num_limbs):
            state = self.input_state[:,i]
            Z = state[..., self.attr_fixed_dim:self.attr_fixed_dim+self.z_num*3]
            Z = Z.contiguous().view(self.batch_size, -1,3).transpose(-2, -1)
            Z_all.append(Z)
        Z_all = torch.stack(Z_all,-3)
        # print("Z_all",Z_all.shape)
        # print("h_all",h_all.shape)
        # print(frame)
        # print(frame.transpose(-2, -1))
        # print("Z_all", Z_all)
        Z_all = torch.einsum('bij,bljk->blik', frame.transpose(-2, -1), Z_all)
        # print("rot Z_all", Z_all)
        # print("Z_all_frame",Z_all.shape)
        Z_all = Z_all.transpose(-2, -1).contiguous().view(Z_all.shape[0], Z_all.shape[1], -1)
        # print("Z_all_frame",Z_all.shape)
        self.input_state[..., self.attr_fixed_dim:self.attr_fixed_dim+self.z_num*3] = Z_all
        # self.input_state = torch.cat([Z_all,h_all],-1)
        # self.input_state
        # print("self.input_state",self.input_state.shape)
        if len(state.shape) == 2:
            self.input_state = self.input_state.squeeze(0)
        return self.input_state
        # print("self.input_state",self.input_state.shape)

        # self.action = self.trans(self.input_state, self.graph)
        # self.action = self.max_action * torch.tanh(self.action)

        # because of the permutation of the states, we need to unpermute the actions now so that the actions are (batch,actions)

        # print("self.action", self.action.shape)
        # self.action = self.action.permute(1, 0, 2)
        # self.action = self.action.contiguous().view(self.action.shape[0], -1)
        # # print("self.action", self.action)

        # if mode == "inference":
        #     self.batch_size = temp
        # return self.action


    def bottom_up_transmission(self, node):

        if node < 0:
            return torch.zeros(
                (self.batch_size, 3, self.msg_dim), requires_grad=True
            ), torch.zeros(
                (self.batch_size, self.msg_dim), requires_grad=True
            )

        if self.msg_up_Z[node] is not None:
            return self.msg_up_Z[node], self.msg_up_h[node]

        state = self.input_state[:,node]
        h_a, Z, h = state[...,:self.attr_fixed_dim], state[..., self.attr_fixed_dim:self.attr_fixed_dim+self.z_num*3], state[..., self.attr_fixed_dim+self.z_num*3:]
        h = torch.cat([h_a, h], dim=-1)
        Z = Z.contiguous().view(self.batch_size, -1,3).transpose(-2, -1)
        # self.gdir = Z[...,1:3]
        # print("shape",Z.shape, h.shape, self.gdir.shape)

        # rad = math.pi/4
        # theta = math.pi*rad
        # O = torch.tensor([[math.cos(theta), -math.sin(theta), 0],
        #             [math.sin(theta), math.cos(theta), 0],
        #             [0, 0, 1]]).unsqueeze(0)
        # O = O.repeat(Z.shape[0],1,1)
        # Z = torch.einsum('bij,bjk->bik', O, Z)


        children = [i for i, x in enumerate(self.parents) if x == node]
        assert (self.max_children - len(children)) >= 0
        children += [-1] * (self.max_children - len(children))
        msg_in_Z = [None] * self.max_children
        msg_in_h = [None] * self.max_children
        for i in range(self.max_children):
            msg_in_Z[i], msg_in_h[i] = self.bottom_up_transmission(children[i])


        self.msg_up_Z[node], self.msg_up_h[node] = self.sNet[node](Z, h, msg_in_Z, msg_in_h)

        return self.msg_up_Z[node], self.msg_up_h[node]

    def top_down_transmission(self, node):
        if node < 0:
            return torch.zeros(
                (self.batch_size, 3, self.msg_dim * self.max_children),
                requires_grad=True,
            ), torch.zeros(
                (self.batch_size, self.msg_dim * self.max_children),
                requires_grad=True,
            )
        elif self.msg_down_Z[node] is not None:
            return self.msg_down_Z[node], self.msg_down_h[node]


        parent_msg_Z, parent_msg_h = self.top_down_transmission(self.parents[node])
        # print("parent_msg",parent_msg.shape)

        # find self children index (first child of parent, second child of parent, etc)
        # by finding the number of previous occurences of parent index in the list
        self_children_idx = self.parents[:node].count(self.parents[node])

        # if the structure is flipped, flip message order at the root
        if self.parents[0] == -2 and node == 1:
            self_children_idx = (self.max_children - 1) - self_children_idx

        msg_in_Z, msg_in_h = self.msg_slice(parent_msg_Z, parent_msg_h, self_children_idx)

        self.frame[node], self.msg_down_Z[node], self.msg_down_h[node] = self.actor[node](self.msg_up_Z[node], self.msg_up_h[node], msg_in_Z, msg_in_h)

        return self.msg_down_Z[node], self.msg_down_h[node]



    def msg_slice(self, x, y, idx):
        return torch.split(x, x.shape[-1] // self.max_children, dim=-1)[idx], torch.split(y, y.shape[-1] // self.max_children, dim=-1)[idx]

    def clear_buffer(self):
        self.msg_down_Z = [None] * self.num_limbs
        self.msg_down_h = [None] * self.num_limbs
        self.msg_up_Z = [None] * self.num_limbs
        self.msg_up_h = [None] * self.num_limbs
        self.Z_h = [None] * self.num_limbs
        self.frame = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs


    def construct_adj_matrix(self):
        adj_matrix = torch.zeros((self.num_limbs, self.num_limbs), dtype=torch.int)
        for i in range(self.edge_index.shape[1]):
            u, v = self.edge_index[:, i]
            adj_matrix[u, v] = 1
            adj_matrix[v, u] = 1  # Since the graph is undirected
        return adj_matrix
    
    def construct_parents(self):
        parents = [-1 for _ in range(self.num_limbs)]
        visited = [False] * self.num_limbs

        def dfs(node, parent):
            visited[node] = True
            if parent is not None:
                parents[node]=parent
            for neighbor in range(self.num_limbs):
                if self.adj_matrix[node, neighbor] == 1 and not visited[neighbor]:
                    dfs(neighbor, node)

        dfs(0, None)  # Start DFS from the root node (node 0)
        return parents
    
    def change_morphology(self, edge_index, num_nodes):
        self.edge_index = edge_index
        self.num_limbs = num_nodes.item()
        self.adj_matrix = self.construct_adj_matrix()
        self.parents = self.construct_parents()
        self.msg_down_Z = [None] * self.num_limbs
        self.msg_down_h = [None] * self.num_limbs
        self.msg_up_Z = [None] * self.num_limbs
        self.msg_up_h = [None] * self.num_limbs
        self.Z_h = [None] * self.num_limbs
        self.frame = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
        self.sNet = nn.ModuleList([self.sNet[0]] * self.num_limbs)
        self.actor = nn.ModuleList([self.actor[0]] * self.num_limbs)

