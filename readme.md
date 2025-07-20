# Unsupervised Sequence Compression via Reversible Memory States

This project implements a simple experimental architecture for compressing sequences into a latent memory state without backpropagation through time (BPTT). It uses a deterministic encoder and decoder, and is evaluated on the byte-level enwik8 dataset.

## Overview

The core idea is to maintain a memory state `h_t` that is updated step-by-step via an encoder:

```
h_{t+1} = f(h_t, x_t)
```

At each step, a decoder attempts to reconstruct both the previous memory state and the current input:

```
(h_t, x_t) ≈ g(h_{t+1})
```

This effectively trains the system to compress the input sequence into the evolving memory state. Since gradients are computed and applied at each step independently, this avoids the need for BPTT.

## Dataset: enwik8

* The dataset is downloaded automatically (10MB slice).
* Byte-level modeling (256 classes).

## Architecture

### Encoder

* Input: one-hot vector of byte `x_t` and memory state `h_t`
* Output: next memory state `h_{t+1}`
* Implemented as a 2-layer feedforward network

### Decoder

* Input: current memory state `h_{t+1}`
* Output: predicted previous memory `\hat{h}_t` and input byte `\hat{x}_t`
* Implemented as two heads:

  * MSE regression for memory state
  * Softmax classifier for byte prediction

## Training

* Per-timestep gradient updates (no sequence unrolling)
* One optimizer step per time step
* Loss is the sum of:

  * Mean squared error (MSE) between `h_t` and `\hat{h}_t`
  * Cross-entropy loss for `x_t`

## Usage

Run the training script:

```bash
python train.py
```

## Future Work

* Replace deterministic decoder with probabilistic Gaussian decoder
* Minimize information gain between `h_t` and `h_{t+1}` to regularize memory usage
* Measure mutual information between memory states
* Evaluate byte-level compression efficiency over longer sequences
