import torch
import torch.nn as nn
import math

from config import ModelConfig

class InputEmbeddings(nn.Module):
    def __init__(self, config: ModelConfig, vocab_size: int) -> None:
        super().__init__()
        self.d_model = config.D_MODEL
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, self.d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, config: ModelConfig, seq_len: int) -> None:
        super().__init__()
        self.d_model = config.D_MODEL
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p=config.DROPOUT)
        
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, self.d_model)
        # Create a vector of shape (seq_len, 1) each value represents the position of the word inside the sentence
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) 
        # Create the denominator of the formula
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))

        #Apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term) #for every word, start at zero and go to the end incrementing by 2
        pe[:, 1::2] = torch.cos(position * div_term) 

        #add a batch dimension so we can apply it to a batch of sentences
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        # Register the positional encoding as a buffer so it gets saved in the state_dict but not as a learned parameter
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        #add the pe to all batches, but only include the positional encodings up to seq_len, but include the entire embedding 
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        #Use parameter so that these are learnable 
        self.alpha = nn.Parameter(torch.ones(1)) # acts as a scalar
        self.bias = nn.Parameter(torch.zeros(1)) # added. aka beta

    def forward(self, x):
        #usually mean cancels the dimension to which the mean is applied, but we want to keep it
        
        mean = x.mean(dim = -1, keepdim=True) 
        std = x.std(dim = -1, keepdim= True)
        return self.alpha * (x - mean) / torch.sqrt(std**2 + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(config.D_MODEL, config.D_FF) #W1 and B1
        self.dropout = nn.Dropout(config.DROPOUT)
        self.linear_2 = nn.Linear(config.D_FF, config.D_MODEL) #W2 and B2
    
    def forward(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.d_model = config.D_MODEL
        self.h = config.NUM_HEADS
        self.d_q_k = config.D_Q_K
        self.d_v = config.D_V

        assert self.d_model % self.h == 0, "d_model is not divisible by h"

        # self.d_k = d_model // h 

        self.w_q = nn.Linear(self.d_model, self.d_q_k * self.h, bias=False) # Wq
        self.w_k = nn.Linear(self.d_model, self.d_q_k * self.h, bias=False) # Wk
        self.w_v = nn.Linear(self.d_model, self.d_v * self.h, bias=False) # Wv

        self.w_o = nn.Linear(self.d_v * self.h, self.d_model) # Wo
        self.dropout = nn.Dropout(config.DROPOUT)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        #we will pass the attention_scores back for visualization
        return (attention_scores @ value), attention_scores
    
    def split_heads(self, x, d):
        # We split the embedding dimension
        # we transpose so that each head will see the entire sequence
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        batch_size, seq_length, _ = x.size()
        #view splits the embedding dimension 
        return x.view(batch_size, seq_length, self.h, d).transpose(1, 2)
    
    def combine_heads(self, x):
        batch_size, _, seq_length, _ = x.size()
        #(batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) -->(batch, seq_len, d_model)
        #In order for view to work, we need to put the tensor in contiguous memory so that it can be done in place
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_v * self.h)
        
    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = self.split_heads(query, self.d_q_k)
        key = self.split_heads(key, self.d_q_k)
        value = self.split_heads(value, self.d_v)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        x = self.combine_heads(x)

        return self.w_o(x)
    
class ResidualConnection(nn.Module):

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.dropout = nn.Dropout(config.DROPOUT)
        self.norm = LayerNormalization()
    
    def forward(self, x, sublayer):
        #can switch the norm and sublayer order 
        return x + self.dropout(self.norm(sublayer(x)))
    

class EncoderBlock(nn.Module):

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.self_attention_block = MultiHeadAttentionBlock(config)
        self.feed_forward_block = FeedForwardBlock(config)
        self.residual_connections = nn.ModuleList([ResidualConnection(config) for _ in range(2)])
    
    def forward(self, x, src_mask):
        # we need the src mask to prevent the padding words from interacting with the other words and vis versa 
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(config) for _ in range(config.NUM_ENCODER_LAYERS)])
        self.norm = LayerNormalization()
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.is_decoder_only = config.NUM_ENCODER_LAYERS == 0
        self.self_attention_block = MultiHeadAttentionBlock(config)
        if not self.is_decoder_only:
            self.cross_attention_block = MultiHeadAttentionBlock(config)
        self.feed_forward_block = FeedForwardBlock(config)
        
        num_connections = 3
        if self.is_decoder_only:
            num_connections = 2
        self.residual_connections = nn.ModuleList([ResidualConnection(config) for _ in range(num_connections)])
    
    #src mask is the mask applied to the encoder while the tgt mask is applied to the decoder values
    def forward(self, x, tgt_mask, encoder_output=None, src_mask=None):
        i = 0
        x = self.residual_connections[i](x, lambda x: self. self_attention_block(x, x, x, tgt_mask))
        i += 1
        if not self.is_decoder_only:
            x = self.residual_connections[i](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
            i += 1
        x = self.residual_connections[i](x, self.feed_forward_block)
        return x 

class Decoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(config) for _ in range(config.NUM_DECODER_LAYERS)])
        self.norm = LayerNormalization()
    
    def forward(self, x, tgt_mask, encoder_output=None, src_mask=None):
        for layer in self.layers:
            x = layer(x, tgt_mask, encoder_output, src_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):

    def __init__(self, config: ModelConfig, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(config.D_MODEL, vocab_size)
    
    def forward(self, x):
        #(batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)

class Transformer(nn.Module):

    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, max_seq_len: int, 
                 config: ModelConfig) -> None:
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.src_embed = InputEmbeddings(config, src_vocab_size)
        self.tgt_embed = InputEmbeddings(config, tgt_vocab_size)
        self.pos = PositionalEncoding(config, max_seq_len)
        self.projection_layer = ProjectionLayer(config, tgt_vocab_size)
        
        #TODO: make the initialization configurable
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(self, src, src_mask):
        
        src = self.src_embed(src)
        src = self.pos(src)
        
        return self.encoder(src, src_mask)

    def decode(self, tgt, tgt_mask, encoder_output=None, src_mask=None):
        
        tgt = self.tgt_embed(tgt)
        tgt = self.pos(tgt)
        
        return self.decoder(tgt, tgt_mask, encoder_output, src_mask)   
    
    def project(self, x):
        return self.projection_layer(x)
    
    def forward(self, tgt, tgt_mask, src=None, src_mask=None):
        x = None
        if src is not None:
            x = self.encode(src, src_mask)
        x = self.decode(tgt, tgt_mask, x, src_mask)
        
        return self.project(x)
    