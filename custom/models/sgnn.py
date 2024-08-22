import torch
import torch.nn as nn
# from .utils import get_fully_connected, SGNNMessagePassingNetwork
from torch_scatter import scatter
import torch.nn.functional as F



# class DistanceRBF:
#     def __init__(self, num_channels=64, start=0.0, stop=2.0):
#         self.num_channels = num_channels
#         self.start = start
#         self.stop = stop * 10
#         self.params = self.initialize_params()

#     def initialize_params(self):
#         offset = torch.linspace(self.start, self.stop, self.num_channels - 2).to('cuda')
#         coeff = (-0.5 / (offset[1] - offset[0])**2).to('cuda')
#         return {'offset': offset, 'coeff': coeff}

#     def __call__(self, dist, dim):
#         assert dist.shape[dim] == 1
#         dist = dist * 10
#         offset_shape = [1] * len(dist.shape)
#         offset_shape[dim] = -1

#         offset = self.params['offset'].reshape(offset_shape).to('cuda')
#         coeff = self.params['coeff']

#         overflow_symb = (dist >= self.stop).type(torch.float32).to('cuda')
#         underflow_symb = (dist < self.start).type(torch.float32).to('cuda')
#         y = dist - offset
#         y = torch.exp(coeff * torch.square(y))
#         return torch.cat([underflow_symb, y, overflow_symb], dim=dim)


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


class SGNNMessagePassingLayer(nn.Module):
    def __init__(self, node_f_dim, node_s_dim, edge_f_dim, edge_s_dim, hidden_dim, vector_dim, activation):
        super(SGNNMessagePassingLayer, self).__init__()
        # self.node_f_dim, self.node_s_dim, self.edge_f_dim, self.edge_s_dim = node_f_dim, node_s_dim, edge_f_dim, edge_s_dim
        # self.hidden_dim = hidden_dim
        # self.activation = activation
        # self.net = BaseMLP(input_dim=(node_f_dim * 2 + edge_f_dim) ** 2 + node_s_dim * 2 + edge_s_dim,
        #                    hidden_dim=hidden_dim,
        #                    output_dim=(node_f_dim * 2 + edge_f_dim) * node_f_dim + node_s_dim,
        #                    activation=activation,
        #                    residual=False,
        #                    last_act=False,
        #                    flat=False)
        # self.self_net = BaseMLP(input_dim=(node_f_dim * 2) ** 2 + node_s_dim * 2,
        #                         hidden_dim=hidden_dim,
        #                         output_dim=node_f_dim * 2 * node_f_dim + node_s_dim,
        #                         activation=activation,
        #                         residual=False,
        #                         last_act=False,
        #                         flat=False)
        self.edge_s = BaseMLP(input_dim=node_s_dim*2+edge_s_dim,
                           hidden_dim=hidden_dim,
                           output_dim=hidden_dim,
                           activation=activation,
                           residual=False,
                           last_act=False,
                           flat=False)
        self.s_mlp = BaseMLP(input_dim=hidden_dim*2 + vector_dim*vector_dim,
                                hidden_dim=hidden_dim,
                                output_dim=hidden_dim,
                                activation=activation,
                                residual=False,
                                last_act=False,
                                flat=False)
        self.edge_mlp = BaseMLP(input_dim=hidden_dim,
                                hidden_dim=hidden_dim,
                                output_dim=vector_dim,
                                activation=activation,
                                residual=False,
                                last_act=False,
                                flat=False)
        self.sf_mlp = BaseMLP(input_dim=hidden_dim*2 + vector_dim*vector_dim,
                                hidden_dim=hidden_dim,
                                output_dim=vector_dim,
                                activation=activation,
                                residual=False,
                                last_act=False,
                                flat=False)
        self.edge_f =  nn.Linear(node_f_dim*2+edge_f_dim, vector_dim, bias=False)
        # linen.Dense(self.vector_dim,# if self.gravity_axis is None else 31,
        #                 kernel_init=self.kernel_init,
        #                 use_bias=False)
        self.f_mlp = nn.Linear(node_f_dim+vector_dim, vector_dim, bias=False)
        # linen.Dense(self.vector_dim,# if self.gravity_axis is None else 31,
        #                 kernel_init=self.kernel_init,
        #                 use_bias=False)

    def forward(self, f, s, edge_index, edge_f=None, edge_s=None):
        _s = torch.cat((s[edge_index[0]], s[edge_index[1]]), dim=-1)
        if edge_s is not None:
                _s = torch.cat((_s, edge_s), dim=-1)  # [M, 2S]
        _f = torch.cat((f[edge_index[0]], f[edge_index[1]]), dim=-1)
        if edge_f is not None:
            _f = torch.cat((_f, edge_f), dim=-1)  # [M, 3, 2F+Fe]
        _f = self.edge_f(_f)   # [M, 3, 31 or self.vector_dim]
        # f2s = jnp.einsum('bij,bjk->bik', jnp.swapaxes(_f, -1, -2), _f)  # [M, vector_dim, vector_dim]
        # f2s = f2s.reshape(f2s.shape[0], -1) 
        # F_norm = jnp.linalg.norm(f2s, dim=-1, keepdims=True) + 1.0
        # _s = torch.cat((_s, f2s), dim=-1)  # [M, 2S]
        _s = self.edge_s(_s)#/F_norm   # [M, 2S]
        s_c = scatter(_s, edge_index[0], dim=0, reduce='mean', dim_size=f.shape[0]) 
        f2s = torch.einsum('bij,bjk->bik', f.transpose(-1, -2), f)  # [M, vector_dim, vector_dim]
        f2s = f2s.reshape(f2s.shape[0], -1) 
        # F_norm = torch.linalg.norm(f2s, dim=-1, keepdims=True) + 1.0
        temp_s = torch.cat((s, s_c, f2s), dim=-1)  # [N, 2S]
        s_out = self.s_mlp(temp_s)#/F_norm

        _f = self.edge_mlp(_s)[:,None,:]*_f
        f_c = scatter(_f, edge_index[0], dim=0, reduce='mean', dim_size=f.shape[0]) 
        temp_f = torch.cat((f, f_c), dim=-1)  # [N, 3, 2vector_dim]
        f_out = self.sf_mlp(temp_s)[:,None,:]*self.f_mlp(temp_f)#/F_norm[:,None,:]


        # if edge_index.shape[1] == 0:
        #     f_c, s_c = torch.zeros_like(f), torch.zeros_like(s)
        # else:
        #     _f = torch.cat((f[edge_index[0]], f[edge_index[1]]), dim=-1)
        #     if edge_f is not None:
        #         _f = torch.cat((_f, edge_f), dim=-1)  # [M, 3, 2F+Fe]
        #     _s = torch.cat((s[edge_index[0]], s[edge_index[1]]), dim=-1)
        #     if edge_s is not None:
        #         _s = torch.cat((_s, edge_s), dim=-1)  # [M, 2S]
        #     _f_T = _f.transpose(-1, -2)
        #     f2s = torch.einsum('bij,bjk->bik', _f_T, _f)  # [M, (2F+Fe), (2F+Fe)]
        #     f2s = f2s.reshape(f2s.shape[0], -1)  # [M, (2F+Fe)*(2F+Fe)]
        #     f2s = F.normalize(f2s, p=2, dim=-1)
        #     f2s = torch.cat((f2s, _s), dim=-1)  # [M, (2F+Fe)*(2F+Fe)+2S+Se]
        #     c = self.net(f2s)  # [M, (2F+Fe)*F+H]
        #     # c = scatter(c, edge_index[0], dim=0, reduce='mean', dim_size=f.shape[0])  # [N, (2F+Fe)*F+H]
        #     f_c, s_c = c[..., :-self.hidden_dim], c[..., -self.hidden_dim:]  # [M, (2F+Fe)*F], [M, H]
        #     f_c = f_c.reshape(f_c.shape[0], _f.shape[-1], -1)  # [M, 2F+Fe, F]
        #     f_c = torch.einsum('bij,bjk->bik', _f, f_c)  # [M, 3, F]
        #     f_c = scatter(f_c, edge_index[0], dim=0, reduce='mean', dim_size=f.shape[0])  # [N, 3, F]
        #     s_c = scatter(s_c, edge_index[0], dim=0, reduce='mean', dim_size=f.shape[0])  # [N, H]
        # # aggregate f_c and f
        # temp_f = torch.cat((f, f_c), dim=-1)  # [N, 3, 2F]
        # temp_f_T = temp_f.transpose(-1, -2)  # [N, 2F, 3]
        # temp_f2s = torch.einsum('bij,bjk->bik', temp_f_T, temp_f)  # [N, 2F, 2F]
        # temp_f2s = temp_f2s.reshape(temp_f2s.shape[0], -1)  # [N, 2F*2F]
        # temp_f2s = F.normalize(temp_f2s, p=2, dim=-1)
        # temp_f2s = torch.cat((temp_f2s, s, s_c), dim=-1)  # [N, 2F*2F+2S]
        # temp_c = self.self_net(temp_f2s)  # [N, 2F*F+H]
        # temp_f_c, temp_s_c = temp_c[..., :-self.hidden_dim], temp_c[..., -self.hidden_dim:]  # [N, 2F*F], [N, H]
        # temp_f_c = temp_f_c.reshape(temp_f_c.shape[0], temp_f.shape[-1], -1)  # [N, 2F, F]
        # temp_f_c = torch.einsum('bij,bjk->bik', temp_f, temp_f_c)  # [N, 3, F]
        # f_out = temp_f_c
        # s_out = temp_s_c
        return f_out, s_out



class SGNN(nn.Module):
    def __init__(self, state_dim, attr_fixed_dim, attr_design_dim, msg_dim=128, p_step=4, activation=nn.SiLU()):
        super(SGNN, self).__init__()
        # initialize the networks
        self.p_step = p_step
        self.state_dim = state_dim
        self.attr_fixed_dim = attr_fixed_dim
        self.attr_design_dim = attr_design_dim
        self.z_num = 6
        self.z_dim = 16
        self.embedding_in = nn.Linear(self.attr_fixed_dim+self.attr_design_dim+self.state_dim-self.z_num*3, msg_dim)
        self.embedding_z = nn.Linear(self.z_num, self.z_dim, bias=False)
        self.embedding_u = nn.Linear(self.z_dim, 1, bias=False)
        self.embedding_out = BaseMLP(input_dim=msg_dim,#+self.z_num*3,
                           hidden_dim=msg_dim,
                           output_dim=self.attr_fixed_dim+self.attr_design_dim+self.state_dim,
                           activation=activation,
                           residual=False,
                           last_act=False,
                           flat=False)
        
        self.message_passing = SGNNMessagePassingLayer(node_f_dim=self.z_dim, node_s_dim=msg_dim, edge_f_dim=self.z_num, edge_s_dim=1, hidden_dim=msg_dim, vector_dim=self.z_dim, activation=activation)
        # self.dist_rbf = DistanceRBF(num_channels=self.z_dim)

    def forward(self, x, edge_index, num_nodes_cum):
        h_a, Z, h = x[...,:self.attr_fixed_dim], x[..., self.attr_fixed_dim:self.attr_fixed_dim+self.z_num*3], x[..., self.attr_fixed_dim+self.z_num*3:]
        h = torch.cat([h_a, h], dim=-1)
        Z = Z.reshape(-1, self.z_num, 3)

        Z = Z.transpose(-2, -1)

        # import math
        # rad = math.pi/4
        # theta = math.pi*rad
        # O = torch.tensor([[math.cos(theta), -math.sin(theta), 0],
        #             [math.sin(theta), math.cos(theta), 0],
        #             [0, 0, 1]]).unsqueeze(0)
        # O = O.repeat(Z.shape[0],1,1)
        # Z = torch.einsum('bij,bjk->bik', O, Z)

        f_p = Z[...,0]  # [M, 3]


        Z0 = Z
        h0 = h

        edge_attr_inter_f = (Z[edge_index[1]] - Z[edge_index[0]]) 
        # edge_attr_inter_s = torch.linalg.norm(edge_attr_inter_f, dim=-1).unsqueeze(-1)
        # edge_attr_inter_f = edge_attr_inter_f.unsqueeze(-1)  # [M, 3, 1]

        edge_attr_inter_s = torch.linalg.norm(f_p[edge_index[1]] - f_p[edge_index[0]], dim=-1).unsqueeze(-1)
        # edge_attr_inter_s = self.dist_rbf(dist, dim=-1) 

        h = self.embedding_in(h)  # [N, H]
        Z = self.embedding_z(Z)

        for _ in range(self.p_step):
            Z, h = self.message_passing(Z, h, edge_index, edge_attr_inter_f, edge_attr_inter_s)  # [N_obj, 3, vector_dim], [N_obj, 2vector_dim]


        # theta = -math.pi*rad
        # O = torch.tensor([[math.cos(theta), -math.sin(theta), 0],
        #             [math.sin(theta), math.cos(theta), 0],
        #             [0, 0, 1]]).unsqueeze(0)
        # O = O.repeat(Z.shape[0],1,1)
        # Z = torch.einsum('bij,bjk->bik', O, Z)

        root_node = []
        start_idx = 0
        for end_idx in num_nodes_cum:
            root_node.extend([start_idx] * (end_idx - start_idx))
            start_idx = end_idx
        root_node = torch.tensor(root_node).to(Z0.device)

        u = self.embedding_u(Z)
        mat = construct_3d_basis_from_1_vectors(u[..., 0])  # [2,3,3]
        f_p = torch.einsum('bij,bjk->bik', mat[root_node].transpose(-1,-2), Z0)  # [N, 3, 32]
        f_p = f_p.transpose(-1, -2)
        f_p = f_p.reshape(f_p.shape[0], -1)  # [N, 3*32]
        # F_norm = torch.linalg.norm(f_p, axis=-1, keepdims=True) + 1.0
        h = torch.cat((f_p, h0), axis=-1)  # [3*N=32*3+H] #s_o_[self.obj_id]


        # h = self.embedding_out(h)
        return h
        # Z = self.embedding_f(jnp.swapaxes(Z, 0, -1))
        # f_p = jnp.swapaxes(f_p, 0, -1)
        # u = self.embedding_u(f_p)
        # mat = construct_3d_basis_from_1_vectors(u[..., 0]) #[2,3,3]
        # f_p = jnp.einsum('bij,bjk->bik', mat.transpose(0,2,1), f_o)  # [N,  3, 32]
        # f_p = f_p.reshape(f_p.shape[0],-1)  # [N, 3*32]
        # # F_norm = jnp.linalg.norm(f_p, axis=-1, keepdims=True) + 1.0
        # ret = torch.cat((f_p, s_p), axis=-1)  # [3*N=32*3+H] #s_o_[self.obj_id]

        # s_p = self.embedding(h_p)  # [N, H]
        # f_o = scatter(torch.stack((x_p, v_p), dim=-1), obj_id, dim=0, reduce='mean')  # [N_obj, 3, x]
        # s_o = scatter(s_p, obj_id, dim=0) * self.eta  # [N_obj, H]
        # f_p = torch.stack((x_p, v_p), dim=-1)  # [N, 3, 2]
        # f_p = torch.cat((f_p - f_o[obj_id], v_p.unsqueeze(-1)), dim=-1)  # [N, 3, 3]
        # s_p = torch.cat((s_o[obj_id], s_p), dim=-1)  # [N, 2H]
        # s_p = self.embedding1(s_p)  # [N, H]
        # edge_attr_inter_f = (x_p[edge_index_inter[0]] - x_p[edge_index_inter[1]]).unsqueeze(-1)  # [M_out, 3, 1]
        # edge_attr_f, edge_attr_s = self.local_interaction(f_p, s_p, edge_index_inter, edge_attr_inter_f)
        # # [M_out, H], [M_out, 3, x]
        # if self.gravity_axis is not None:
        #     g_o = torch.zeros_like(f_o)[..., 0]  # [N_obj, 3]
        #     g_o[..., self.gravity_axis] = 1
        #     g_o = g_o * self.obj_g_p(s_o)
        #     f_o = torch.cat((f_o, g_o.unsqueeze(-1)), dim=-1)  # [N_obj, 3, x+1]

        # num_obj = obj_id.max() + 1  # N_obj
        # edge_index_o = get_fully_connected(num_obj, device=obj_id.device, loop=True)  # [2, M_obj]
        # edge_mapping = obj_id[edge_index_inter[0]] * num_obj + obj_id[edge_index_inter[1]]  # [M_out]
        # edge_attr_o_f = scatter(edge_attr_f, edge_mapping, dim=0, reduce='mean', dim_size=num_obj ** 2)  # [M_obj, 3, x]
        # edge_attr_o_s = scatter(edge_attr_s, edge_mapping, dim=0, reduce='mean', dim_size=num_obj ** 2)  # [M_obj, H]
        # edge_pseudo = torch.ones(edge_attr_s.shape[0]).to(edge_attr_s.device)  # [M_, 1]
        # count = scatter(edge_pseudo, edge_mapping, dim=0, reduce='sum', dim_size=num_obj ** 2)
        # mask = count > 0
        # edge_index_o, edge_attr_o_f, edge_attr_o_s = edge_index_o[..., mask], edge_attr_o_f[mask], edge_attr_o_s[mask]
        # f_o_, s_o_ = self.object_message_passing(f_o[..., 1:], s_o, edge_index_o, edge_attr_o_f, edge_attr_o_s)  # [N_obj, 3, 2]

        # edge_attr_inner_f = (x_p[edge_index_inner[0]] - x_p[edge_index_inner[1]]).unsqueeze(-1)  # [M_in, 3, 1]
        # f_p_ = torch.cat((f_o_[obj_id], f_p), dim=-1)
        # s_p_ = torch.cat((s_o_[obj_id], s_p), dim=-1)
        # s_p_ = self.embedding2(s_p_)  # [N, H]
        # if self.gravity_axis is not None:
        #     g_p = torch.zeros_like(f_p_)[..., 0]  # [N, 3]
        #     g_p[..., self.gravity_axis] = 1
        #     g_p = g_p * self.particle_g_p(s_p_)
        #     f_p_ = torch.cat((f_p_, g_p.unsqueeze(-1)), dim=-1)  # [N_obj, 3, x+1]
        # f_p_, s_p_ = self.object_to_particle(f_p_, s_p_, edge_index_inner, edge_attr_inner_f)  # [N, 3, x], [N, H]
        # v_out = f_p_[..., 0]
        # return v_out



