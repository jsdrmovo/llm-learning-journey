import math, collections

def perplexity_char(text: str):
    # Build bigram counts with Laplace smoothing
    text = text.strip("\n")
    if not text: return float("nan")
    counts = collections.defaultdict(lambda: collections.Counter())
    vocab = set(text)
    for a, b in zip(text[:-1], text[1:]):
        counts[a][b] += 1
        vocab.add(a); vocab.add(b)
    V = len(vocab)
    logp = 0.0; n = 0
    for a, b in zip(text[:-1], text[1:]):
        c_ab = counts[a][b]
        c_a = sum(counts[a].values())
        p = (c_ab + 1) / (c_a + V)  # Laplace
        logp += -math.log(p + 1e-12)
        n += 1
    return math.exp(logp / max(n,1))

def mc_accuracy(pred_indices, gold_indices):
    assert len(pred_indices) == len(gold_indices)
    correct = sum(int(p==g) for p,g in zip(pred_indices, gold_indices))
    return correct / max(1, len(gold_indices))

if __name__ == "__main__":
    text = "hello hello world! minimal eval."
    ppl = perplexity_char(text)
    print(f"Bigram PPL (toy): {ppl:.2f}")

    # toy multiple-choice
    gold = [0,1,2,1,0]
    # trivial baseline: always choose 0
    pred = [0]*len(gold)
    acc = mc_accuracy(pred, gold)
    print(f"MC accuracy (always-0 baseline): {acc:.2f}")
