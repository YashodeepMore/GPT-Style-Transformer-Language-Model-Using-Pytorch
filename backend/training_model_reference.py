
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 10


batch_size = 32
block_size = 128

max_iters = 10000
eval_interval = 500

learning_rate = 5e-4

n_embd = 256
n_head = 4
n_layer = 4

dropout = 0.2




class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd,head_size, bias=False)
        self.query = nn.Linear(n_embd,head_size, bias=False)
        self.value = nn.Linear(n_embd,head_size, bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))

        self.dropout = nn.Dropout(dropout)



    def forward(self,x):

        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        head_size = k.size(-1)
        wei = q @ k.transpose(-2,-1) * head_size**-0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei,dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v

        return out


class MulltiHeadAttention(nn.Module):
    def __init__(self, num_head, head_size):
        super().__init__()

        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    # feed forward network : simple linear layer followed by non linear layer
    def __init__(self,n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self,n_embd, n_head):
        super().__init__()

        head_size = n_embd // n_head
        self.sa = MulltiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x





class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding_tabel = nn.Embedding(vocab_size,n_embd)
        self.possisional_encoding_tabel = nn.Embedding(block_size,n_embd)
        # self.sa_head = MulltiHeadAttention(4, n_embd//4)
        # self.ffwd = FeedForward(n_embd)

        # self.blocks = nn.Sequential(
        #     Block(n_embd,n_head=4),
        #     Block(n_embd,n_head=4),
        #     Block(n_embd,n_head=4),
        # )
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head) for _ in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd,vocab_size)

    def forward(self,idx,targets = None):
        B,T = idx.shape

        token_embedding = self.token_embedding_tabel(idx)
        possision_emb = self.possisional_encoding_tabel(torch.arange(T, device=device))

        x = token_embedding + possision_emb
        # x = self.sa_head(x)
        # x = self.ffwd(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None

        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_token):

        for _ in range(max_new_token):

            #croping idx to last block size tokens
            idx_cond = idx[:, -block_size:]

            #getting the predictions
            logits, loss = self(idx_cond)

            #focus only on last time stamp
            logits = logits[:,-1,:] # become (B,C)

            #appling softmax for get probalities
            prob = F.softmax(logits,dim=-1)

            #sample from distribuyion
            idx_next = torch.multinomial(prob, num_samples=1)

            idx = torch.cat((idx, idx_next), dim =1)

            # print(itos[idx_next.item()], end='')
        return idx




model = LanguageModel().to(device)
best_val_loss = float('inf')

optimizers = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()

        print(
            f"step {iter}: "
            f"train loss {losses['train']:.4f} "
            f"val loss {losses['val']:.4f}"
        )

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']

            # torch.save({
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizers.state_dict(),
            #     'iteration': iter,
            #     'val_loss': best_val_loss,
            # }, 'best_model.pt')

            # print(f"Saved checkpoint at iteration {iter}")

    xb,yb = get_batch('train')

    logits, loss = model(xb,yb)

    optimizers.zero_grad(set_to_none=True)
    loss.backward()
    optimizers.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context,max_new_token=500)[0].tolist()))



