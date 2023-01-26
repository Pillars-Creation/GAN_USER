import torch

from model.layers import FactorizationMachine, MultiLayerPerceptron

class WideAndDeep(torch.nn.Module):

    def __init__(self, description, embed_dim, mlp_dims, dropout, user_id_name='user_id'):
        super().__init__()
        # assert len(description) == 11, 'unillegal format of {}'.format(description)
        self.features = [name for name, _, type in description if type != 'label']
        assert user_id_name in self.features, 'unkown user id name'
        self.description = {name: (size, type) for name, size, type in description}
        self.user_id_name = user_id_name
        self.build(embed_dim, mlp_dims, dropout)
    
    def build(self, embed_dim, mlp_dims, dropout):
        self.emb_layer = torch.nn.ModuleDict()
        self.ctn_emb_layer = torch.nn.ParameterDict()
        self.ctn_linear_layer = torch.nn.ModuleDict()
        embed_output_dim = 0
        for name, (size, type) in self.description.items():
            if type == 'spr':
                self.emb_layer[name] = torch.nn.Embedding(size, embed_dim)
                embed_output_dim += embed_dim
            elif type == 'ctn':
                self.ctn_linear_layer[name] = torch.nn.Linear(1, 1, bias=False)
            elif type == 'seq':
                self.emb_layer[name] = torch.nn.Embedding(size, embed_dim)
                embed_output_dim += embed_dim
            elif type == 'label':
                pass
            else:
                raise ValueError('unkown feature type: {}'.format(type))
        self.mlp = MultiLayerPerceptron(embed_output_dim, mlp_dims, dropout)
        return

    def only_optimize_userid(self):
        for name, param in self.named_parameters():
            if self.user_id_name not in name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)
        return

    def init(self):
        for param in self.parameters():
            torch.nn.init.uniform_(param, -0.01, 0.01)

    def optimize_all(self):
        for name, param in self.named_parameters():
            param.requires_grad_(True)
        return

    def forward(self, x_dict):
        user_id_emb = self.emb_layer[self.user_id_name](x_dict[self.user_id_name])
        loss = self.forward_with_user_id_emb(user_id_emb, x_dict)
        return loss

    def forward_with_user_id_emb(self, user_id_emb, x_dict):
        linears = []
        embs = []
        for name, (_, type) in self.description.items():
            if name == self.user_id_name or type == 'label':
                continue
            x = x_dict[name]
            if type == 'spr':
                embs.append(self.emb_layer[name](x).squeeze(1))
            elif type == 'ctn':
                linears.append(self.ctn_linear_layer[name](x))
            elif type == 'seq':
                embs.append(self.emb_layer[name](x).sum(dim=1))
            else:
                raise ValueError('unkwon feature: {}'.format(name))
        emb = torch.concat([user_id_emb.squeeze(1)] + embs, dim=1)
        linear_part = torch.concat(linears, dim=1).sum(dim=1, keepdims=True)
        res = (linear_part + self.mlp(emb)).squeeze(1)
        return torch.sigmoid(res)

