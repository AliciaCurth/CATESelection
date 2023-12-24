# CATESelection
Sklearn-style implementations of model selection criteria for CATE estimation.

This repo contains code to replicate the results presented in the ICML23 paper ['In Search of Insights, Not Magic Bullets: Towards Demystification of the Model Selection Dilemma in Heterogeneous Treatment Effect Estimation'](https://arxiv.org/abs/2302.02923).

## Running the experiments
To run the experiments, first install the requirements:
```bash
pip install -r requirements.txt
```

Then the experiments can be run using the following command:
```bash
python run_experiments.py --setup <setup>
```
where `<setup>` is one of `A, B, C, D`.

Generating the figures can be done with [`experiments/notebooks/Results-plots-main-paper.ipynb`](experiments/notebooks/Results-plots-main-paper.ipynb).

## Citing

If you use this software please cite the corresponding paper:

```
@inproceedings{curth2023search,
  title={In Search of Insights, Not Magic Bullets: Towards Demystification of the Model Selection Dilemma in Heterogeneous Treatment Effect Estimation},
  author={Curth, Alicia and van der Schaar, Mihaela},
  booktitle={International Conference on Machine Learning},
  year={2023},
  organization={PMLR}
}
```
