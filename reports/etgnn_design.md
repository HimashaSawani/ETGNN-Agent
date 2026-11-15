# Preliminary ETGNN design

## Dataset constraint
Elliptic edges connect transactions within the same time step.
There are no observed edges between different time steps.

## Proposed comparison
Use GAT as the closest graph baseline. ETGNN will add an explicit
representation of the transaction's time step. Compare GAT and ETGNN
using the same node features, graph snapshots, training masks, and
evaluation metrics.

## Temporal mechanism to implement and test
Use a fixed sinusoidal encoding calculated from the transaction time step, concatenate it with the 165 node features, and apply graph attention within each available snapshot. This encoding can be calculated for unseen validation and test time steps.

## Required ablation
Compare GAT without the time embedding against ETGNN with it.
This tests whether the explicit time representation adds value.

## Limits
Time-step embeddings alone do not establish evolving links between
transactions across time steps. Do not claim that the dataset contains
cross-time-step edges.
