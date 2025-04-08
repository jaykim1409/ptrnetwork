import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class PtrAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.encoder_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.decoder_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)
        
    def forward(self, enc_outputs, dec_out, pointer_mask):
        enc_proj = self.encoder_proj(enc_outputs)  # (batch_size, input_len, hidden_dim)
        dec_proj = self.decoder_proj(dec_out).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        u = self.v(torch.tanh(enc_proj + dec_proj))  # (batch_size, input_len, 1)
        u = u.squeeze(-1)
        
        u = u.masked_fill(pointer_mask == 0, float('-inf'))
        log_probs = F.log_softmax(u, dim=-1)
        
        return log_probs
        
        
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        
    def forward(self, inputs):
                
        enc_outputs, _ = self.lstm(inputs)  # (batch_size, input_len, hidden_dim*2)
        
        return enc_outputs
    
    
class Decoder(nn.Module):
    def __init__(self, hidden_dim, attention):
        super().__init__()
        self.lstm = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)
        self.attention = attention
        
    def forward(self, decoder_input, decoder_hidden, enc_outputs, pointer_mask):
        dec_out, decoder_hidden = self.lstm(decoder_input, decoder_hidden)  # (batch_size, 1, hidden_dim)
        dec_out = dec_out.squeeze(1)    # (batch_size, hidden_dim)
        
        log_probs = self.attention(enc_outputs, dec_out, pointer_mask)
        
        return log_probs
    
    
    
class PtrNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # self.hidden_dim = hidden_dim
        self.encoder = Encoder(input_dim, hidden_dim)
        self.attention = PtrAttention(hidden_dim)
        self.decoder = Decoder(hidden_dim, self.attention)
        
    # def forward(self, inputs, target_len, padding_mask=None, targets=None):
    def forward(self, inputs, target_len, targets=None):
        batch_size, input_len, _ = inputs.size()
        
        # Encoder
        enc_outputs = self.encoder(inputs)  # (batch_size, input_len, hidden_dim*2)
        
        # Init decoder state
        decoder_input = enc_outputs.mean(dim=1, keepdim=True)  # start token (average of enc)
        decoder_hidden = None

        pointer_logits = []
        pointer_mask = torch.ones(batch_size, input_len).to(inputs.device)

        loss = 0.0
        
        for t in range(target_len):
            # Decoder - Attention step
            log_probs = self.decoder(decoder_input, decoder_hidden, enc_outputs, pointer_mask)
            pointer_logits.append(log_probs)
            
            # Training (Teaching Force) : compute loss
            if targets is not None:
                target_t = targets[:, t]  # (batch_size,)
                loss_t = F.nll_loss(log_probs, target_t, reduction='mean')
                loss += loss_t

                # Update pointer mask to avoid duplicates
                pointer_mask = pointer_mask.scatter(1, target_t.unsqueeze(1), 0)

                # Update decoder input (teacher forcing)
                decoder_input = enc_outputs.gather(1, target_t.view(batch_size, 1, 1).expand(-1, -1, enc_outputs.size(2)))

            else:
                # Inference: greedy selection
                selected = log_probs.argmax(dim=-1)
                # print(selected.size())
                pointer_mask = pointer_mask.scatter(1, selected.unsqueeze(1), 0)
                decoder_input = enc_outputs.gather(1, selected.view(batch_size, 1, 1).expand(-1, -1, enc_outputs.size(2)))

        if targets is not None:
            loss = loss / target_len
            return loss
        else:
            # Return pointer predictions (during inference)
            return torch.stack(pointer_logits, dim=1).exp().argmax(dim=-1)  # (batch_size, target_len)
