RCP-DTA: Representation-Aware Conformal Prediction for Reliable Uncertainty Quantification in Drug-Target Affinity

📖 Introduction
Predicting Drug-Target Affinity (DTA) is crucial for drug discovery, but point estimates alone lack the reliability needed for clinical decision-making. RCP-DTA introduces a novel framework that integrates multi-modal representation learning with Conformal Prediction (CP).

By leveraging deep structural and sequential features of both drugs and proteins, RCP-DTA provides adaptive prediction intervals that guarantee statistical coverage, offering a robust measure of uncertainty for every prediction.

Key Features：
  ·Multi-Modal Encoding: Utilizes MolGNN for drug molecular graphs and ProGNN (with Virtual Feedback) for protein structures.

  ·Deep Feature Fusion: Employs UniCrossAttention for inter-modal interaction and SSFusion (Gate Fusion) for refined representation.

  ·Reliable Uncertainty: Implements Local Quantile Mapping to generate sample-specific confidence intervals.
