# Where your project overlaps with the course

Checked against the 9 lecture files you gave me: Sessions 01, 02, 03, 04, 05, 06, 07, 08 and 14.
Limit of this check: the Transformer-specific parts (positional encoding, residual + LayerNorm layers) are probably taught in the next session after 08, which was not in the files. I cannot confirm that overlap.

## Summary

| Session | Topic | Overlap with your project |
|---|---|---|
| 01 | Intro to deep learning | Yes — foundations (neurons, activations, backprop, RNN as a network type) |
| 02 | Activations, perceptron, forward/backward | Yes — activations in every model, the training loop |
| 03 | Loss, optimizers, over/underfitting | **Strong** — cross-entropy, Adam, learning rate, schedule, curves |
| 04 | Initialization and regularization | **Strong** — dropout, weight decay, early stopping (initialization not explored) |
| 05 | CNNs | Indirect — ideas only (weight sharing, inductive bias, skip connections, parameter counting). No CNN is used |
| 06 | Computer vision, transfer learning | Indirect — transfer learning idea (MediaPipe), augmentation label rule. No detection or segmentation |
| 07 | Sequence models | **Core** — RNN, LSTM, GRU, many-to-one, vanishing gradient, splitting rules |
| 08 | Embeddings, seq2seq, attention | **Strong for attention** (your Transformer). Seq2seq, teacher forcing, BLEU not used |
| 14 | Types of GANs | None |

---

## Session 01 — Introduction

| Topic | Where in your project | Why it overlaps | Purpose of the topic |
|---|---|---|---|
| Deep learning learns features end to end | The sequence models learn from raw landmark numbers; no hand-designed motion features | You never wrote rules like "hand moves up" | Removes the human feature-engineering bottleneck |
| Neuron, layers, activation function | `nn.Linear` + ReLU head in all three models; gates use sigmoid/tanh | Every model is stacked layers of weighted sums plus non-linearity | Non-linearity lets a network learn more than straight-line boundaries |
| Softmax output | Last layer gives 11 scores; softmax turns them into probabilities (`evaluate.py`, `predict.py`) | Multi-class output | Gives probabilities that sum to 1 |
| Backpropagation: forward then backward | The loop in `train.py` | You run exactly the 6-step algorithm from the slide | How a network learns from its error |
| Types of networks (feedforward, RNN, CNN) | LSTM and GRU are RNNs; the Dense head is feedforward | You compare recurrent models with an attention model | Choosing the architecture that fits the data type |
| Cons: needs a lot of data | Only 230 training clips | Your small dataset is the textbook weakness | Tells you what to be careful about |

## Session 02 — Activations and perceptron

| Topic | Where | Why it overlaps | Purpose |
|---|---|---|---|
| ReLU | `Dense 64 + ReLU` in every model; the Transformer's feed-forward layer | Direct use | Fast, cheap non-linearity (its variants fix "dying" neurons; you did not check for that) |
| MLP and why hidden layers matter | The Dense 64 head on top of the recurrent/attention part | A small MLP classifier | A hidden layer bends the decision boundary |
| Forward / backward propagation, training loop | `train.py` (`zero_grad`, forward, `loss.backward()`, `optimizer.step()`) | Same five steps as slide 15 | The recipe that trains every network |
| PyTorch `nn.Module` vs scikit-learn | Your models are `nn.Module`; scikit-learn is used only for splitting, metrics and the logistic-regression baselines | The comparison table in the session | PyTorch gives full control and GPU; the assignment requires it |

## Session 03 — Loss, optimization, overfitting (strong)

| Topic | Where | Why it overlaps | Purpose |
|---|---|---|---|
| Cross-entropy with softmax | `nn.CrossEntropyLoss` in `train.py`; models return raw scores (logits) | You avoid the "softmax twice" bug from the pitfalls slide | Penalises confident mistakes; the right loss for multi-class classification |
| Debug tip: starting loss ≈ −log(1/classes) | First-epoch training loss was 2.35–2.40 and −log(1/11) = 2.40 | The check passes | Quick sign that the setup is correct |
| Adam, learning rate 1e-3 | `train.py` | Exactly the recommended starting point | Per-weight adaptive steps; a default that "just works" |
| Learning rate too small / right / too large | The tuning search: GRU at 1e-4 is too slow (82–86%); LSTM at 1e-3 is too high (48% vs 70%) | Your results show the slide's picture | The most important hyper-parameter to tune |
| Learning rate schedule | `ReduceLROnPlateau` (halve after 10 flat epochs) | Direct use | Big steps early, small steps near the minimum |
| Epoch, batch, iteration | Batch 16, 230 clips → 15 iterations per epoch, up to 200 epochs | The vocabulary slide | Describes a training run precisely |
| Overfitting and underfitting; watch validation loss | Curves slide: LSTM underfits (train ≈ 40%); Transformer's validation loss rises while training loss falls | Both patterns appear in your three models | Diagnose before choosing a fix |
| Prevention rules (early stopping, simpler model, regularization, more data) | Early stopping, dropout, weight decay used; more data is your future work | Listed on the slide | The toolkit for generalisation |

## Session 04 — Initialization and regularization (strong, with gaps)

| Topic | Where | Why it overlaps | Purpose |
|---|---|---|---|
| Dropout (0.2) and `model.train()` / `model.eval()` | All models; `evaluate.py`, `predict.py` call `eval()` | Direct use, and you avoid "the #1 bug" | Random switching-off of units acts like a cheap ensemble |
| L2 weight decay | `weight_decay` in the tuning grid (0 vs 1e-4) | You tested it as the assignment asks | Shrinks weights so none are used only to memorise noise. Result: no effect on GRU |
| Early stopping, return the *saved best* weights | Patience 25; the best-by-validation checkpoint is what you test | Same rule: keep the best weights, not the last | Nearly free regularization |
| "Perfect training performance is a warning sign" | GRU reached 100% on train and test | The slide's caution applies, so you ran extra checks (`diagnostics.py`) | Reminder to question a perfect score |
| Regularization trades training error for generalization | Transformer: dropout 0.2, training accuracy 100%, validation 94% | Your Discussion slide | Definition to use in Q&A |
| Chain of multiplications, vanishing/exploding signals | Possible reason the LSTM struggles over 90 steps (a hypothesis, not tested) | The same idea reappears in Session 07 | Why initialization and gates matter |
| **Not covered in your project:** Xavier/He initialization, data augmentation, multi-task learning, L1 | You left PyTorch's default initialization; no augmentation | These are the gaps | Good "future work" items |

## Session 05 — CNNs (indirect)

| Topic | Where | Why it overlaps | Purpose |
|---|---|---|---|
| Weight sharing / "hard prior" (inductive bias) | An RNN reuses one cell at every frame; the Transformer reuses the same attention at every position | Same argument as the kernel shared over an image, applied to time | Fewer parameters and better generalisation. Use this when you say "inductive bias" |
| Counting parameters | You report trainable parameters; GRU ≈ ¾ of LSTM (3 vs 4 weight blocks) | The "body-count" slide | Compare model size honestly |
| Skip connections (ResNet) | Gates in LSTM/GRU and the residual connections inside the Transformer layers play the same role | Session 07 makes this link explicitly | Give gradients a short path so deep/long chains still train |
| Transfer learning with a pretrained backbone | MediaPipe is a pretrained, frozen feature extractor | The same "borrow someone else's features" idea, but not one of your trained approaches | A strong start when data is scarce |
| **Not used:** convolution, pooling, AlexNet/VGG/ResNet | — | A 1D CNN would be a valid 4th approach (see Session 07's decision table) | — |

## Session 06 — Computer vision (indirect)

| Topic | Where | Why it overlaps | Purpose |
|---|---|---|---|
| Transfer learning as a dial (freeze all ↔ fine-tune all) | MediaPipe sits at "freeze all"; your three models are "from scratch" | The assignment's second dimension, *training strategy*, is unused | A future 4th approach could use a pretrained model |
| Augmentation must preserve the label | Your planned "flip with left/right hand swap" | The slide warns that flips break tasks where left and right differ | Avoid training on corrupted labels |
| Domain gap | One signer, one room | Same worry: does it work on a different person or camera? | Explains why test accuracy can overstate real performance |
| **Not used:** detection (YOLO, IoU, NMS), segmentation | MediaPipe does pose/hand detection inside, but you do not train it | — | — |

## Session 07 — Sequence models (core)

| Topic | Where | Why it overlaps | Purpose |
|---|---|---|---|
| Order carries meaning; many-to-one task | Each clip (90 frames) → one label, using the last hidden state (`h_n[-1]` in `lstm_model.py`, `gru_model.py`) | The "classifying a whole sequence" row of the decision table | Match the architecture to the input/output shape |
| RNN: one cell shared over time, parameters independent of length | Your recurrent layers run 90 steps with the same weights | Slide 14 of the session | Handles any length without more weights |
| Backprop through time and the vanishing gradient | Possible factor in the LSTM's weak result (hypothesis) | The reason gates exist | Explains forgetting over long ranges |
| LSTM (3 gates + cell) and GRU (2 gates) | `lstm_model.py`, `gru_model.py`: identical layout, only the cell changes | The "one-word change" slide | GRU is lighter; LSTM is heavier |
| Decision table: short sequences, little data → GRU | GRU won with 230 training clips | The course's own guidance matches your result | Use it as the explanation, with the caveats from `diagnostics.py` |
| Time-series splitting rule: never leak the future | Your split is random and stratified; clips were recorded back-to-back | Same risk in a different form: near-duplicate clips on both sides | Keeps the test score honest. This is a limitation to state |
| Windowing, text/embedding pipeline | Each 90-frame clip is a window; the Transformer's `Linear 258→128` plays the role an embedding plays for words | Landmarks are numbers, so no token lookup is needed | Turn raw data into the size the model expects |
| 1D CNN for local patterns | Not used | The table lists it as the choice when only local patterns matter | A cheap extra baseline |

## Session 08 — Embeddings, encoder–decoder, attention (strong for attention)

| Topic | Where | Why it overlaps | Purpose |
|---|---|---|---|
| Scaled dot-product attention, softmax(QKᵀ/√d)V | `nn.TransformerEncoderLayer` in `transformer_model.py` | PyTorch implements exactly this | Lets every frame read every other frame |
| Multi-head attention (4 heads) | `nhead=4`, `d_model=128` | Directly the "run attention several times in parallel" slide | Different heads can follow different relationships |
| "Where RNN runs out": recurrence is sequential | Your Transformer has no recurrence | The next-session idea you implemented | Parallel processing and short paths between frames. At your size all three train in 6–8 s, so you cannot show a speed benefit |
| Pretrained embeddings (word2vec, GloVe) | Not used | Would be a "training strategy" change | An option for text; not for landmarks |
| **Not used:** encoder–decoder, teacher forcing, beam search, BLEU | — | You classify, you do not generate | Only needed if you later turn signs into sentences |

## Session 14 — GANs

No overlap. A conditional GAN could in theory generate synthetic landmark clips to enlarge the dataset, but that is a speculative idea, not part of the project.

---

## How to use this

For the Discussion slide and Q&A, the assignment asks you to use course concepts. Map them like this:

| Concept | Where it applies in your project |
|---|---|
| Capacity | Transformer (307k parameters) vs 230 training clips: it overfits slightly; GRU (179k) fits cleanly |
| Data size | 230 training clips, one signer: the main limit on what you can claim |
| Regularization | Dropout 0.2, early stopping, weight decay (no effect on GRU) |
| Inductive bias | Recurrence (read in order, keep memory) vs attention (compare all frames directly) |
| Transfer learning | Not used for the three models; MediaPipe is a borrowed feature extractor; pretrained models are a possible 4th approach |

Gaps worth mentioning as future work, because the lessons cover them and the project does not: weight initialization, data augmentation, a 1D-CNN baseline, a pretrained/transfer-learning approach.
