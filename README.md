# Predicting Drug Dissolution Enhancement with Quantum ML

This is a classical and quantum machine learning project built for the OQI Hackathon 2026. The
challenge asks whether co-milling a crystalline drug with PVP K25 will meaningfully improve its
dissolution rate, and whether that can be predicted from the drug's molecular structure alone.
The full brief is in [`OQI_Hackathon_2026.pdf`](OQI_Hackathon_2026.pdf). The dataset and the
classical benchmark we compare against (Q squared of 0.77) come from
[Patzmann et al. 2024](https://doi.org/10.1016/j.ejps.2024.106780).


## Getting started

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS or Linux

pip install -r requirements.txt

python main.py --stage all
```

That runs the whole pipeline against a local simulator, so no IBM account is needed. It takes
roughly 20 to 30 minutes, mostly spent in the quantum stage, because every circuit is actually
executed rather than solved on paper. Results land in `data/results/` and `plots/`.

To run on real IBM Quantum hardware, copy `.env.example` to `.env`, fill in your IBM Cloud
credentials, run `python scripts/setup_ibm_account.py` once, then pass `--backend ibm-runtime`.
Add `--max-samples 8` for a small, cheap check before committing to a full run on a real device.

## The problem

29 crystalline drugs were co-milled with PVP K25 and tested for dissolution. The response
variable is the ratio of how much dissolved with co-milling versus without it. A ratio of 2.0 or
higher counts as a Responder, meaning the treatment at least doubled dissolution. 13 of the 29
drugs are Responders.

We predict this from 14 molecular descriptors computed with RDKit, plus two experimental
measurements already in the dataset: particle size and apparent solubility. The measurements the
target ratio is built from are never allowed as model inputs, since that would just be handing
the model the answer.

## What we built

**Classical models:** an SVC with an RBF kernel, a Random Forest, a Gradient Boosting classifier,
and a PLS regression baseline that reproduces the published benchmark.

**Quantum models:** a quantum kernel SVM, a version of that same kernel where the encoding is
trained to line up with the labels, a variational quantum classifier, and a quantum convolutional
network built the same way a classical CNN is, with convolution and pooling layers that shrink
the qubit count down before a final dense layer. There are regression versions of most of these
too, predicting the dissolution ratio directly instead of just a yes or no answer.

Every model is evaluated the same way: 29 rounds of leave one out cross validation, where each
drug takes a turn being the one held out. Feature scaling and feature selection are both done
fresh on the training data each round, so nothing about the held out drug ever leaks into the
model that predicts it.

## Results

Our best model overall is the trained quantum kernel SVM, which reached 96.6 percent accuracy,
ahead of the best classical model, Gradient Boosting, at 93.1 percent. On the regression side the
result flips: classical PLS reached a Q squared of 0.756, close to the published benchmark of
0.77, while our best quantum regressor reached 0.600. We treat both of those as real, honest
outcomes rather than picking the one that flatters quantum computing.

The comparison plots live in `plots/comparison/`.

## Bonus work

Beyond the required models, we also built a version of the quantum kernel whose weights are
trained specifically to maximize alignment with the labels, a study comparing accuracy under an
ideal simulator against a noisy or real hardware backend, and a blind prediction tool that takes
a raw SMILES string it has never seen and returns a Responder or Non-Responder call with a
confidence score.

## Notebooks

`notebooks/00_project_walkthrough.ipynb` runs through the entire project end to end, and is the 
easiest place to start if you want to see everything work without reading the source files first. 
The other three notebooks are shorter, focused walkthroughs of individual pipeline stages.

## Data source

The experimental dissolution data and the four reference variables used in the source paper come
from Patzmann et al., European Journal of Pharmaceutical Sciences, 2024. Molecular structures
were looked up from PubChem by drug name.
