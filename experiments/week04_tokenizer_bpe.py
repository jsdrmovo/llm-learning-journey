from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from pathlib import Path

CORPUS = [
    "attention is all you need\n",
    "transformers build on attention\n",
    "hello world! llm theory first\n",
]

def main():
    Path("outputs").mkdir(exist_ok=True)
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=200, min_frequency=1, special_tokens=["[PAD]","[UNK]"])
    tok.train_from_iterator(CORPUS, trainer=trainer)
    tok.save("outputs/tokenizer.json")
    # Stats
    lens = [len(tok.encode(s).ids) for s in CORPUS]
    print("Avg tokens:", sum(lens)/len(lens), "Max:", max(lens), "Min:", min(lens))
    print("Saved to outputs/tokenizer.json")

if __name__ == "__main__":
    main()
