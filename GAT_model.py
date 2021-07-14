import torch
import torch.nn as nn


class BaseGatLayer(nn.Module):
    
    head_dimension = 1
    
    def __init__(
        self,
        num_in_features,
        num_out_features,
        num_heads,
        output_activation,
        concat=True,
        dropout_prob=0.5,
        add_skip_connection=True,
        bias=True,
    ):
        super().__init__()
        
        self.add_bias = bias
        self.concat = concat
        self.num_heads = num_heads
        self.num_in_features = num_in_features
        self.num_out_features = num_out_features
        self.add_skip_connection = add_skip_connection
        
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_heads, num_out_features))
        self.linear_proj = nn.Linear(num_in_features, num_heads*num_out_features, bias=False)
        
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)
        
        self.softmax = nn.Softmax(dim=-1)
        self.output_activation = output_activation
        self.dropout = nn.Dropout(p=dropout_prob)
        
        self.init_params()
        
    def init_params(self):
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)
        if self.add_bias:
            nn.init.zeros_(self.bias)
        
    def skip_concat(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()
            
        if self.add_skip_connection:
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]: # F_IN = F_OUT
                # cast input to (N, 1, F_IN) and auto broadcast to (N, N_H, F_IN) 
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else: # F_IN != F_OUT
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_heads, self.num_out_features)
                
        if self.concat:
            # shape = (N, N_H, F_OUT) -> (N, N_H*F_OUT)
            out_nodes_features = out_nodes_features.view(-1, self.num_heads * self.num_out_features)
        else:
            # shape = (N, N_H, F_OUT) -> (N, F_OUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dimension)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.output_activation is None else self.output_activation(out_nodes_features)
    
    
class GatLayer(BaseGatLayer):
    
    def __init__(
        self,
        num_in_features,
        num_out_features,
        num_heads,
        output_activation,
        concat=True,
        dropout_prob=0.5,
        add_skip_connection=True,
        bias=True,
    ):
        super().__init__(
            num_in_features,
            num_out_features,
            num_heads,
            output_activation,
            concat=concat,
            dropout_prob=dropout_prob,
            add_skip_connection=add_skip_connection,
            bias=bias,
        )
        
    
    def forward(self, input_data):
        in_nodes_features, connectivity_mask = input_data
        num_of_nodes = in_nodes_features.shape[0]
        
        assert_string = 'Expected connectivity matrix with shape=({0},{0}), got shape={1}.'.format(
            num_of_nodes,
            connectivity_mask.shape,
        )
        assert connectivity_mask.shape == (num_of_nodes, num_of_nodes), assert_string
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_heads, self.num_out_features)
        nodes_features_proj = self.dropout(nodes_features_proj)
        nodes_features_proj = nn.LeakyReLU(0.1)(nodes_features_proj)
        
        scores_source = torch.sum((nodes_features_proj * self.scoring_fn_source), dim=-1, keepdim=True)
        scores_target = torch.sum((nodes_features_proj * self.scoring_fn_target), dim=-1, keepdim=True)
        
        scores_source = scores_source.transpose(0, 1)
        scores_target = scores_target.permute(1, 2, 0)
        
        all_scores = scores_source + scores_target
        
        all_attention_coefficients = self.softmax(all_scores + connectivity_mask)
        
        

        out_nodes_features = torch.bmm(all_attention_coefficients, nodes_features_proj.transpose(0, 1))

        out_nodes_features = out_nodes_features.permute(1, 0, 2)

        out_nodes_features = self.skip_concat(all_attention_coefficients, in_nodes_features, out_nodes_features)
        
        return (out_nodes_features, connectivity_mask)
    
    
class GAT(nn.Module):
    def __init__(
        self,
        num_of_layers,
        num_heads_per_layer,
        num_features_per_layer,
        add_skip_connection=True,
        bias=True,
        dropout=0.6,
    ):
        super().__init__()
        
        assert num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1
        
        num_heads_per_layer = [1] + num_heads_per_layer
        
        gat_layers = []
        for i in range(num_of_layers):
            layer = GatLayer(
                num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],
                num_out_features=num_features_per_layer[i+1],
                num_heads=num_heads_per_layer[i+1],
                concat=True if i < num_of_layers - 1 else False,
                output_activation=nn.ELU() if i < num_of_layers - 1 else None,
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
            )
            gat_layers.append(layer)
        self.gat = nn.Sequential(*gat_layers)
        
    def forward(self, x):
        return self.gat(x)