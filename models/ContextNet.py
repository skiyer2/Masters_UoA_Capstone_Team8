import torch
import torch.nn as nn
from .basemodel import BaseModel
from .layers.base_mlp import MLP
from .layers.base_liner import Linear
from .layers.base_context import ContextualEmbedding, ContextNetBlock

class SessionContextNet(BaseModel):

    def __init__(self, fix_SparseFeat, fix_DenseFeat, args):
        super(SessionContextNet, self).__init__(fix_SparseFeat, fix_DenseFeat, args)

        _ = self.get_input_dim()
        _ = self.get_field_index()

        self.block_num = args.model_params['block_num']
        self.use_ce_every_layer = args.model_params['use_ce_every_layer']
        emb_dim = args.model_params['embedding_dim']

        if self.use_ce_every_layer:
            self.ce = nn.ModuleList([
                ContextualEmbedding(
                    num_fields=len(self.field_index),
                    feature_dim=emb_dim,
                    increase=args.model_params['increase_ratio'],
                ) for _ in range(self.block_num)
            ])
            self.cn = nn.ModuleList([
                ContextNetBlock(
                    num_fields=len(self.field_index),
                    feature_dim=emb_dim,
                    net_model=args.model_params['net_model']
                ) for _ in range(self.block_num)
            ])
        else:
            self.ce = ContextualEmbedding(
                num_fields=len(self.field_index),
                feature_dim=emb_dim,
                increase=args.model_params['increase_ratio'],
            )
            self.cn = nn.ModuleList([
                ContextNetBlock(
                    num_fields=len(self.field_index),
                    feature_dim=emb_dim,
                    net_model=args.model_params['net_model']
                ) for _ in range(self.block_num)
            ])
        self.add_regularization_net_weight(self.ce.parameters())
        self.add_regularization_net_weight(self.cn.parameters())

        self.gru = nn.GRU(
            input_size=len(self.field_index) * emb_dim,
            hidden_size=args.model_params.get('gru_hidden_dim', 64),
            batch_first=True,
            num_layers=args.model_params.get('gru_layers', 1)
        )

        self.dnn_linear = nn.Linear(
            args.model_params.get('gru_hidden_dim', 64), 1, bias=False
        )
        self.add_regularization_net_weight(self.dnn_linear.parameters())

        self.raw_input_dim = args.model_params.get("raw_input_dim") 
        if self.raw_input_dim is None:
            self.raw_input_dim = len(fix_SparseFeat) + len(fix_DenseFeat)

        if args.model_params.get('use_linear', False):
            self.linear = Linear(
                self.raw_input_dim,
                dropout=args.model_params['linear_dropout'],
                use_bias=True
            )
            self.add_regularization_net_weight(self.linear.parameters())

        if args.model_params['loss'] == 'binary_crossentropy':
            self.prediction_layer = nn.Sequential(
                nn.Sigmoid()
            )
        self.add_regularization_net_weight(self.prediction_layer.parameters())

    def forward(self, x):
        batch_size, session_len, feat_dim = x.shape

        contextualized_seq = []
        for t in range(session_len):
            xt = x[:, t, :] 
            sparse_dict, dense_dict = self.get_embeddings(xt)
            all_emb = [values for res, values in sparse_dict.items()]
            all_emb += [values for res, values in dense_dict.items()]
            all_emb = torch.cat(all_emb, dim=-1) 
            fm_emb = [values.unsqueeze(1) for res, values in sparse_dict.items()]
            fm_emb = torch.cat(fm_emb, dim=1) 

            if self.use_ce_every_layer:
                out = fm_emb
                for idx in range(self.block_num):
                    out = self.ce[idx](out)
                    out = self.cn[idx](out)
            else:
                out = self.ce(fm_emb)
                for idx in range(self.block_num):
                    out = self.cn[idx](out)
            out_flat = out.reshape(batch_size, -1)
            contextualized_seq.append(out_flat.unsqueeze(1)) 

        contextualized_seq = torch.cat(contextualized_seq, dim=1)

        gru_out, _ = self.gru(contextualized_seq)

        logits = self.dnn_linear(gru_out).squeeze(-1)

        if self.args.model_params.get('use_linear', False):
            flat_input = x.view(batch_size * session_len, feat_dim).float()
            linear_logit = self.linear(flat_input).view(batch_size, session_len)
            logits += linear_logit

        y_pred = self.prediction_layer(logits)
        reg_loss = self.get_regularization_loss()
        return y_pred, reg_loss
