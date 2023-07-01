"""
Create by Juwei Yue on 2020-3-26
IKTN model
"""

from utils import *
from base import *
import torch.optim as optim
import random
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
class IKTN(Base):
    def __init__(self, positional_size, n_heads, n_layers, *args, **kwargs):
        super(IKTN, self).__init__(*args, **kwargs)
        lr=1e-3
        weight_decay=5e-4
        self.positional_size = positional_size

        self.arg_self_attention = SelfAttention(self.embedding_size, n_heads[0], self.dropout)
        self.layer_norm = nn.LayerNorm(self.embedding_size)
        self.event_composition = EventComposition(self.embedding_size, self.hidden_size, self.dropout)
        self.gcn = GCN(self.hidden_size, n_layers, self.dropout)
        self.mlp = MLP(self.dropout)
        self.attention = Attention(self.hidden_size)
        self.lstm = lstm(self.dropout)
        self.mlp_optimizer = optim.Adamax(self.mlp.parameters(), weight_decay=weight_decay)
        self.reconstruct_optimizer = optim.Adamax(self.lstm.parameters())
        self.optimizer = optim.Adamax([{"params": self.arg_self_attention.parameters()},
                                     {"params": self.layer_norm.parameters()},
                                     {"params": self.event_composition.parameters()},
                                     {"params": self.attention.parameters()},
                                     {"params": self.gcn.parameters()},
                                     {"params": self.embedding.parameters(), "lr":lr * 0.06}],
                                    lr=lr, weight_decay=weight_decay)


    def adjust_event_chain_embedding(self, embedding):
        """
        shape: (batch_size, 52, embedding_size) -> (batch_size * 5, 36, embedding_size)
            52: (8 context events + 5 candidate events) * 4 arguments
            36: (8 context events + 1 candidate events) * 4 arguments
        """
        embedding = torch.cat(tuple([embedding[:, i::13, :] for i in range(13)]), 1)
        context_embedding = embedding[:, 0:32, :].repeat(1, 5, 1).view(-1, 32, self.embedding_size)
        candidate_embedding = embedding[:, 32:52, :].contiguous().view(-1, 4, self.embedding_size)

        event_chain_embedding = torch.cat((context_embedding, candidate_embedding), 1)

        return event_chain_embedding

    def adjust_event_embedding(self, embedding):
        """
        shape: (batch_size * 5, 9, hidden_size) -> (batch_size, 13, hidden_size)
        """
        embedding = embedding.view(embedding.size(0) // 5, -1, self.hidden_size)
        context_embedding = torch.zeros(embedding.size(0), 8, self.hidden_size).to(device)
        for i in range(0, 45, 9):
            context_embedding += embedding[:, i:i+8, :]
        context_embedding /= 8.0
        candidate_embedding = embedding[:, 8::9, :]
        event_embedding = torch.cat((context_embedding, candidate_embedding), 1)
        return event_embedding

    def forward(self, inputs, matrix, label):
        # embedding layer

        inputs_embed = self.embedding(inputs)
        inputs_embed = self.adjust_event_chain_embedding(inputs_embed)

        # argument attention layer
        mask = compute_mask(self.positional_size)
        arg_embed = self.arg_self_attention(inputs_embed, mask)
        arg_embed = self.layer_norm(torch.add(inputs_embed, arg_embed))

        # event composition layer
        event_embed = self.event_composition(arg_embed)
        event_embed = self.adjust_event_embedding(event_embed)

        # MLP layer
        context = event_embed[:, 0:8, :]
        candidate = event_embed[:, 8:13, :]
        result = torch.unsqueeze(torch.cat((context[0, 7, :], candidate[0, label[0], :]), dim=0), 0)
        for k in range(1, context.shape[0]):
            tamp = torch.unsqueeze(torch.cat((context[k, 7, :], candidate[k, label[k], :]), dim=0), 0)
            result = torch.cat((result, tamp), 0)
        for k in range(0, context.shape[0]):
            a=random.randint(0, 6)
            b=a+1
            tamp = torch.unsqueeze(torch.cat((context[k, a, :], context[k, b, :]), dim=0), 0)
            result = torch.cat((result, tamp), 0)
        domain_label = torch.cat((torch.full((context.shape[0],), 0), torch.full((context.shape[0],), 1)), 0)
        indices = torch.randperm(result.size(0))
        result_shuffled = result[indices]
        domain_label = domain_label[indices].to(device)
        mlp_outputs = self.mlp(result_shuffled)

        gcn_outputs = self.gcn(event_embed, matrix)
        lstm_out = self.lstm(gcn_outputs)





        # attention layer
        h_i, h_c = self.attention(gcn_outputs)

        # score functions
        outputs = -torch.norm(h_i - h_c, 2, 1).view(-1, 5)

        return outputs, mlp_outputs, domain_label, gcn_outputs, lstm_out, event_embed
