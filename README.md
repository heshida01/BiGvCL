# BiGvCL

PyTorch code for multi-class drug-gene interaction prediction with graph and
contrastive learning.

## Data

Place datasets under `data/drugbank/` or `data/dgidb/`. Each dataset needs
`train.csv` and `test.csv` with:

```text
drug_id,gene_id,label
```

## Usage

```bash
python src/main.py --data drugbank --epoch 500 --gpu 0
python src/main.py --data dgidb --epoch 200 --tstEpoch 1 --gpu 0
```

Use `--gpu -1` for CPU. Checkpoints are saved to
`saved_models/ensemble_models/`.
