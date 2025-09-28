import torch

def dpo_loss(policy_logits, ref_logits, pref_pairs, beta=0.1):
    # policy_logits, ref_logits: (N, V) for N items
    # pref_pairs: list of (i, w, l) : item index i, winner class w, loser class l
    logp = policy_logits.log_softmax(dim=-1)
    logp_ref = ref_logits.log_softmax(dim=-1)
    losses = []
    for i, w, l in pref_pairs:
        term = (logp[i, w] - logp[i, l]) - beta * (logp_ref[i, w] - logp_ref[i, l])
        losses.append(-term)
    return torch.stack(losses).mean()

def accuracy(policy_logits, pref_pairs):
    pred = policy_logits.softmax(-1).argmax(-1)  # (N,)
    ok = 0; total = 0
    for i, w, _ in pref_pairs:
        ok += int(pred[i].item() == w)
        total += 1
    return ok / max(total, 1)

def main():
    torch.manual_seed(0)
    N, V = 8, 5  # 8 items, 5 choices each
    theta = torch.zeros(V, requires_grad=True)  # shared logits over classes
    # reference model slightly biased
    ref = torch.randn(V)

    # Construct synthetic preferences per item
    winners = torch.randint(0, V, (N,))
    losers = (winners + torch.randint(1, V, (N,))) % V
    pairs = [(i, int(winners[i]), int(losers[i])) for i in range(N)]

    opt = torch.optim.Adam([theta], lr=0.2)
    for step in range(200):
        policy_logits = theta.unsqueeze(0).repeat(N, 1)  # same logits per item (toy)
        ref_logits = ref.unsqueeze(0).repeat(N, 1)
        loss = dpo_loss(policy_logits, ref_logits, pairs, beta=0.1)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 40 == 0:
            acc = accuracy(policy_logits.detach(), pairs)
            print(f"step {step:03d} | loss {loss.item():.4f} | pref-acc {acc:.2f}")

    policy_logits = theta.unsqueeze(0).repeat(N, 1)
    print("Final pref-acc:", accuracy(policy_logits, pairs))

if __name__ == "__main__":
    main()