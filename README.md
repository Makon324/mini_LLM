# TinyStories Transformer

A small, from-scratch GPT-style language model implemented in PyTorch. The model is trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset and generates short English children's stories locally.

The project includes:

- `proj3.ipynb` — notebook used for tokenizer training, data preparation, model training, and experiments.
- `inference.py` — command-line script for generating stories with the trained model.
- `tinystories_bpe.json` — trained 4,096-token BPE tokenizer.
- `tinystories_model.pt` — trained model weights.
- `report.md` — detailed project report.

## Requirements

- Python 3.9 or newer
- PyTorch
- Hugging Face Tokenizers
- Hugging Face Datasets
- Jupyter Notebook (needed for training and experimentation)

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Generate a story

Run the inference script from the project directory. The tokenizer and model weights must be present in the same directory as `inference.py`.

```bash
python inference.py --prompt "Once upon a time" --tokens 200 --temp 0.8 --top_k 10
```

The script automatically uses CUDA when available and otherwise falls back to the CPU.

### Command-line options

| Option | Default | Description |
| --- | ---: | --- |
| `--prompt` | `""` | Text to continue. |
| `--tokens` | `200` | Maximum number of new tokens to generate. |
| `--temp` | `0.8` | Sampling temperature; higher values produce more randomness. |
| `--top_k` | `10` | Restricts sampling to the top K most likely tokens. |

Additional examples:

```bash
python inference.py
python inference.py --prompt "One day"
python inference.py --prompt "A little fox found" --tokens 150 --temp 0.7 --top_k 10
```

Generation stops early when the model produces the `<eos>` token or reaches the requested token limit.

## Train or experiment

Open the notebook with:

```bash
jupyter notebook proj3.ipynb
```

The notebook streams the TinyStories dataset, trains the custom BPE tokenizer, prepares 256-token context windows, trains the Transformer, and saves the resulting tokenizer and model weights.

## Model overview

The model is a decoder-only Transformer implemented without `nn.Transformer`. It uses:

- 4 Transformer blocks;
- 8 causal self-attention heads;
- 256-dimensional embeddings;
- a context length of 256 tokens;
- a 4,096-token BPE vocabulary;
- pre-LayerNorm residual blocks and GELU feed-forward layers;
- tied token-embedding and language-model-head weights.
