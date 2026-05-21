# BiGvCL

PyTorch code for multi-class drug-gene interaction prediction with graph and
contrastive learning.

## Data

Place datasets under `data/drugbank/` or `data/dgidb/`. Each dataset needs
`train.csv` and `test.csv` with:

```text
drug_id,gene_id,label
```

## Usage Example

```bash
python src/main.py --data drugbank --epoch 500 --gpu 0

```

## Citation

```bibtex
@article{he2026bigvcl,
  title={BiGvCL: bipartite graph-based cross-domain contrastive learning model for the predicting drug-gene interactions},
  author={He, Shida and others},
  journal={Briefings in Bioinformatics},
  volume={27},
  number={1},
  pages={bbaf710},
  year={2026}
}
```

He, Shida, et al. "BiGvCL: bipartite graph-based cross-domain contrastive learning model for the predicting drug-gene interactions." *Briefings in Bioinformatics* 27.1 (2026): bbaf710.

