from torch import nn
import torch
from nets.graph_layers import MultiHeadEncoder, MLP_for_actor, EmbeddingNet, PositionalEncoding, PositionalEncodingSin
from nets.graph_layers import MLP3
from torch.distributions import Normal, Gamma, Categorical


class mySequential(nn.Sequential):
    def forward(self, inputs, q_length):
        for module in self._modules.values():
            # if type(inputs) == tuple:
            #     inputs = module(*inputs)
            # else:
            #     inputs = module(inputs)
            inputs = module(inputs, q_length)
        return inputs


class MLP(torch.nn.Module):
    def __init__(self,
                 opts,
    ):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(opts.embedding_dim, opts.hidden_dim)
        self.fc2 = torch.nn.Linear(opts.hidden_dim, opts.embedding_dim)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, in_, holder):
        result = self.fc1(in_)
        result = self.ReLU(self.fc2(result).squeeze(-1))
        return result



class Actor(nn.Module):

    def __init__(self,
                 opts, 
                 ):
        super(Actor, self).__init__()

        self.embedding_dim = opts.embedding_dim
        self.hidden_dim = opts.hidden_dim
        self.n_heads_actor = opts.encoder_head_num
        self.decoder_hidden_dim = opts.decoder_hidden_dim        
        self.n_layers = opts.n_encode_layers
        self.normalization = opts.normalization
        self.node_dim = opts.node_dim
        self.op_dim = opts.op_dim
        # self.llm_hidden = llm_hidden
        self.op_embed = opts.op_embed_dim
        self.max_action = opts.maxAct
        self.max_sigma=opts.max_sigma
        self.min_sigma=opts.min_sigma
        self.opts = opts
        # config = [{'in': self.embedding_dim, 'out': self.hidden_dim, 'drop_out': 0.001, 'activation': 'ReLU'},
        #           {'in': self.hidden_dim, 'out': self.hidden_dim, 'drop_out': 0.001, 'activation': 'ReLU'},
        #           {'in': self.hidden_dim, 'out': self.embedding_dim, 'drop_out': 0.001, 'activation': 'ReLU'},
        #          ]
        # self.log_sigma_min = -20.
        # self.log_sigma_max = -1.5
        if self.opts.sep_state:
            self.op_embedder = EmbeddingNet(self.op_dim, self.op_embed)
            self.fla_embedder = EmbeddingNet(self.node_dim, self.op_embed)

            self.embedder = EmbeddingNet(self.op_embed + self.op_embed,
                                         self.embedding_dim)
        else:
            self.embedder = EmbeddingNet(self.node_dim + self.op_dim if self.opts.morphological else self.node_dim,
                                         self.embedding_dim)
        
        if opts.positional is not None and opts.positional != "None":
            self.pos_embedding = PositionalEncoding(self.embedding_dim, opts.maxCom) if opts.positional == 'learnt' else PositionalEncodingSin(self.embedding_dim, opts.maxCom)
        
        if opts.encoder == 'attn':
            self.encoder = mySequential(*(
                    MultiHeadEncoder(self.n_heads_actor,
                                    self.embedding_dim,
                                    self.hidden_dim,
                                    self.normalization)
                    for _ in range(self.n_layers)))  # stack L layers
        else:
            self.encoder = MLP(opts)

        self.decoder = MLP_for_actor(self.embedding_dim, self.decoder_hidden_dim, self.max_action)

        print(self.get_parameter_number())

    def get_parameter_number(self):
        
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Actor: Total': total_num, 'Trainable': trainable_num}

    def forward(self, x_in, q_length=None, detach_state=False, to_critic=False, only_critic=False):
        """
        x_in: shape=[bs, ps, feature_dim]
        """
        # print(x_in)
        if detach_state:
            x_in = x_in.detach()
        if self.opts.sep_state:
            pe_id, x_ind, x_fla = x_in[:, :, 0].long(), x_in[:, :, 1:self.op_dim+1], x_in[:, :, 1+self.op_dim:]
                
            ind_em = self.op_embedder(x_ind)
            fla_em = self.fla_embedder(x_fla)
            h_em = self.embedder(torch.concatenate((ind_em, fla_em), -1))
        else:
            pe_id = None
            h_em = self.embedder(x_in)

        # pass through embedder
        if self.opts.positional is not None and self.opts.positional != "None":
            # h_em = self.pos_embedding(self.embedder(torch.concatenate((llm_em, state), -1)), pe_id)  # [bs, n_comp, dim_em]
            h_em = self.pos_embedding(h_em, pe_id)  # [bs, n_comp, dim_em]
        # else:
        #     # h_em = llm_em
        #     h_em = self.embedder(x_in)
        # pass through encoder
        logits = self.encoder(h_em, q_length)  # [bs, n_comp, dim_em]
        # share embeddings to critic net
        if only_critic:
            return logits
        # pass through decoder
        decoded = (torch.tanh(self.decoder(logits)) + 1.) / 2.
        # print(decoded)

        decoded[:, :, torch.arange(decoded.shape[-1]//2)*2+1] = decoded[:, :, torch.arange(decoded.shape[-1]//2)*2+1] * (self.max_sigma-self.min_sigma)+self.min_sigma

        return (decoded, logits) if to_critic else decoded

        # mu_op = torch.softmax(self.mu_net_op(logits), -1)
        # co_op = torch.softmax(self.co_net_op(logits), -1)

        # policy_mu = Categorical(mu_op)
        # policy_co = Categorical(co_op)

        # sample actions (number of controlled params, bs, ps)
        # if fixed_action is not None:
        #     action = torch.tensor(fixed_action)
        # else:
        #     action = torch.stack([policy_F.sample(), policy_Cr.sample(), policy_mu.sample(), policy_co.sample()])

        # torch.clamp_(action[0], self.F_range[0], self.F_range[1])  # F
        # torch.clamp_(action[1], self.Cr_range[0], self.Cr_range[1])  # Cr

        # softmax (bs, ps * number of controlled params)
        # ll = torch.cat((policy_F.log_prob(action[0]), policy_Cr.log_prob(action[1]), policy_mu.log_prob(action[2]), policy_co.log_prob(action[3])), -1)
        # ll[ll < -1e5] = -1e5

        # if require_entropy:
        #     entropy = torch.cat((policy_F.entropy(), policy_Cr.entropy(), policy_mu.entropy(), policy_co.entropy()), -1)  # for logging only
        #     out = (action,
        #            ll.sum(1),  # bs
        #            logits if to_critic else None,
        #            entropy)
        # else:
        #     out = (action,
        #            ll.sum(1),  # bs
        #            logits if to_critic else None,)

        # return out
    
    def get_logp(self, logits, actions):
        bs = logits.shape[0]
        logp = torch.zeros(bs)
        entor_p = []
        for i in range(bs):  # for each task
            logit = logits[i].cpu()
            action = actions[i]
            for j in range(len(action)):  # for each component
                act = action[j]
                lgt = logit[j]
                for k in range(len(act)):  # for each action
                    policy = Normal(lgt[k*2], lgt[k*2+1])
                    lp = policy.log_prob(act[k])
                    if torch.isinf(lp):
                        lp = 1e-32
                    logp[i] += lp
                    entor_p.append(policy.entropy())
        return logp, entor_p
