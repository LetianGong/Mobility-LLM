from transformers import BertForMaskedLM

from .base import *
from .layers import PositionalEncoding, ContinuousEncoding


def get_batch_mask(B, L, valid_len):
    mask = repeat(torch.arange(end=L, device=valid_len.device),
                  'L -> B L', B=B) < repeat(valid_len, 'B -> B L', L=L)  # (B, L)
    return mask


class BertLM(Encoder):
    def __init__(self, d_model, output_size, bert_path, 
                 dis_feats=[], num_embeds=[], con_feats=[],
                 mlp_grad=True, pooling='mean'):
        super().__init__('BertLM-d{}-o{}-p{}'.format(d_model, output_size, pooling))

        self.output_size = output_size
        self.d_model = d_model
        self.pooling = pooling
        self.dis_feats = dis_feats
        self.con_feats = con_feats

        self.pos_encode = PositionalEncoding(d_model, max_len=2001)
        if len(dis_feats):
            assert len(dis_feats) == len(num_embeds), \
                'length of num_embeds list should be equal to the number of discrete features.'
            self.dis_embeds = nn.ModuleList([nn.Embedding(num_embed, d_model) for num_embed in num_embeds])
        else:
            self.dis_embeds = None

        if len(con_feats):
            self.con_embeds = nn.ModuleList([ContinuousEncoding(d_model) for _ in con_feats])
        else:
            self.con_embeds = None

        self.bert = BertForMaskedLM.from_pretrained(bert_path)
        emb_size = self.bert.config.emb_size
        self.emb_size = emb_size
        self.hidden_size = self.bert.config.hidden_size

        # Froze or Fine-tune the parameters of BERT.
        for i, (name, param) in enumerate(self.bert.named_parameters()):
            if 'ln' in name or 'wpe' in name:  # or 'mlp' in name:
                param.requires_grad = True
            elif 'mlp' in name and mlp_grad:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.seq_projector = nn.Sequential(nn.Linear(d_model, emb_size, bias=False),
                                           nn.LayerNorm(emb_size),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(emb_size, emb_size))
        self.poi_projector = nn.Sequential(nn.Linear(emb_size, emb_size, bias=False),
                                           nn.LayerNorm(emb_size),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(emb_size, emb_size))

        self.out_linear = nn.Sequential(nn.Linear(self.bert.config.hidden_size, output_size, bias=False),
                                        nn.LayerNorm(output_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(output_size, output_size))

        self.trip_mask_token = nn.Parameter(torch.zeros(emb_size).float(), requires_grad=True)
        self.poi_mask_token = nn.Parameter(torch.zeros(emb_size).float(), requires_grad=True)

    def forward(self, trip, valid_len, o_poi_embeddings, d_poi_embeddings,
                trip_mask=None, poi_mask=None, **kwargs):
        B, L, E_in = trip.shape

        trip_batch_mask = get_batch_mask(B, L+2, valid_len+1)
        src_batch_mask = get_batch_mask(B, L+2, valid_len+2)

        h = torch.zeros(B, trip.size(1), self.d_model).to(trip.device)
        if self.dis_embeds is not None:
            for dis_embed, dis_feat in zip(self.dis_embeds, self.dis_feats):
                h += dis_embed(trip[..., dis_feat].long())  # (B, L, E)
        if self.con_embeds is not None:
            for con_embed, con_feat in zip(self.con_embeds, self.con_feats):
                h += con_embed(trip[..., con_feat].float())
        h += self.pos_encode(h)
        h = self.seq_projector(h)

        input_seq = torch.zeros(B, L+2, self.emb_size).to(trip.device)
        input_seq[:, 0] += self.poi_projector(o_poi_embeddings)
        input_seq[:, 1:-1] += h
        input_seq[src_batch_mask.long() - trip_batch_mask.long() == 1] += self.poi_projector(d_poi_embeddings)

        if trip_mask is not None:
            input_seq[:, 1:-1][trip_mask] = self.trip_mask_token
        if poi_mask is not None:
            o_placeholder = torch.zeros(B, L+2).to(trip.device)
            o_placeholder[:, 0] = 1
            o_placeholder = o_placeholder.bool()
            o_poi_mask = repeat(poi_mask[:, 0], 'b -> b l', l=L+2)
            d_poi_mask = repeat(poi_mask[:, 1], 'b -> b l', l=L+2)
            input_seq[o_placeholder & o_poi_mask] = self.poi_mask_token
            input_seq[(src_batch_mask.long() - trip_batch_mask.long() == 1) & d_poi_mask] = self.poi_mask_token

        memory = self.bert(inputs_embeds=input_seq, attention_mask=src_batch_mask, output_hidden_states=True).hidden_states[-1]
        memory = torch.nan_to_num(memory)

        memory = self.out_linear(memory)  # (B, E_out) or (B, L, E_out)

        if bool(kwargs.get('pretrain', False)):
            return memory

        if self.pooling == 'mean':
            mask_expanded = repeat(src_batch_mask.logical_not(), 'B L -> B L E', E=memory.size(2))  # (B, L, E)
            memory = memory.masked_fill(mask_expanded, 0)  # (B, L, E)
            memory = torch.sum(memory, 1) / valid_len.unsqueeze(-1)
        elif self.pooling == 'cls':
            memory = memory[:, 0]

        return memory
