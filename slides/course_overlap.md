# Q&A study guide: course concepts and your code

Use this to prepare for the 5-minute Q&A. It has four parts:

1. Your project in one minute (the story to tell)
2. Where your project uses the lessons, session by session, with ready-to-say answers
3. Your code in plain language, file by file
4. Rapid-fire questions and the numbers to remember

Checked against the lecture files you gave me: Sessions 01–08 and 14. The Transformer-specific details (positional encoding, residual + LayerNorm inside the encoder) are probably taught in the session after 08, which was not in those files, so this guide explains them without a lecture reference.

**How to answer any question:** give the one-sentence answer first, then one piece of evidence from your project ("…for example, my LSTM only reached 40% training accuracy"). If you don't know, say so and say how you would check. Honesty is graded ("accuracy, depth, and honesty of answers").

---

## Part 1 — Your project in one minute

> I recognise 11 Khmer sign words from a webcam. MediaPipe turns each video frame into 258 numbers (body pose plus both hands), and one sign is a clip of 90 frames. I recorded 330 clips myself, split them once into 230 train, 50 validation and 50 test, and trained three sequence models from scratch in PyTorch: an LSTM, a GRU and a Transformer encoder. Same data, same split, same training recipe; only the architecture changes. GRU scored 100% on the test set, the Transformer 98%, the LSTM 38%. The LSTM underfit, and tuning its learning rate lifted it to 70% on validation. My most important finding is a caution: a simple model with no sense of time also scores 100%, and shuffling the frames barely hurts GRU, so my dataset is easy and the results don't prove the models learned movement. The next step is a second signer and signs that differ only in movement.

---

## Part 2 — Where your project uses the lessons

Each topic follows the same pattern: **what the lesson says → where it is in your project → why you did it → the question you may get and an answer you can say.**

### Session 07 — Sequence models (the core of your project)

**1. Order matters; an MLP throws it away.**
- *Lesson:* an MLP sees a flat vector with no idea that frame 3 comes after frame 2. Shuffle every sequence the same way and an MLP scores the same. We need a model that reads one step at a time and carries something forward.
- *Your project:* a sign is movement over 90 frames, so you used sequence models (LSTM, GRU) and attention (Transformer), not a plain MLP.
- *Twist you can mention:* you ran the lesson's shuffle idea as a test. GRU only dropped from 100% to 94% with shuffled frames, so on your data order matters less than you hoped. That is your "caution" slide.
- **Q: "Why not just use a normal neural network (MLP)?"**
  A: "An MLP has no notion of order; Session 07 shows it scores the same on shuffled sequences. Signs are movements, so I chose models that read the frames in order. Interestingly, my shuffle test shows my current dataset doesn't need much order, which is a limitation I report."

**2. The RNN: one cell reused at every time step.**
- *Lesson:* the same weights are used at step 1, 2, 3 … ("weight sharing across time"), so the number of parameters doesn't depend on the sequence length. The hidden state *h* is the memory.
- *Your project:* each LSTM/GRU layer runs over all 90 frames with the same weights. Your 90-frame clips didn't need 90× the parameters.
- **Q: "What is the hidden state?"**
  A: "It's the model's running memory. At each frame it combines the new frame with the old memory. After frame 90, it's a summary of the whole clip, and that's what I classify."

**3. Many-to-one.**
- *Lesson:* three shapes: one→many, many→one, many→many. Classifying a whole sequence is many-to-one: read the final hidden state.
- *Your project:* exactly this. In `lstm_model.py` and `gru_model.py` the line `x = h_n[-1]` takes the memory after the last frame and passes it to the classifier.
- **Q: "Is your task many-to-one or many-to-many?"**
  A: "Many-to-one: 90 frames in, one word out. I use the final hidden state."

**4. Backpropagation through time and the vanishing gradient.**
- *Lesson:* the gradient from the last step back to the first is a product of many terms, so it shrinks to almost nothing (the 0.5⁵⁰ example). A plain RNN forgets long-range links.
- *Your project:* with 90 steps this is a real risk. It's a *possible* reason the LSTM was hard to train, but you didn't test it directly, so call it a hypothesis.
- **Q: "Why did the LSTM fail?"**
  A: "It underfit: only 40% accuracy even on its training data. A model with no sense of time scores 100% on the same data, so the task isn't hard; the LSTM's training just didn't work under these settings. Lowering the learning rate from 0.001 to 0.0001 raised its validation accuracy from 48% to 70%, so the learning rate is a big part of it. Training over 90 steps with little data may also play a role, but I didn't test that."

**5. LSTM vs GRU: gates.**
- *Lesson:* a gate is a valve between 0 and 1 that decides what to keep, forget or output. LSTM has 3 gates and a separate cell state ("conveyor belt"). GRU has 2 gates (update, reset) and no separate cell, so it has fewer weights. The lesson's advice: "GRU: less data, shorter sequences, want faster training. Start here. LSTM: very long dependencies, plenty of data."
- *Your project:* same layout for both (64 → 128 → 64 units). GRU has 178,827 parameters vs the LSTM's 236,811, about three-quarters, because it has 3 weight blocks per layer instead of 4.
- *Why GRU won here:* you have little data (230 training clips), which is the case where the lesson says to start with GRU.
- **Q: "What's the difference between LSTM and GRU, and why did GRU win?"**
  A: "Both use gates to protect memory. LSTM has three gates and a separate cell state; GRU merges them into two gates, so it has about a quarter fewer parameters. With only 230 training clips, the smaller, simpler model was easier to train, which matches the lesson's advice to start with GRU when data is limited."

**6. Gates are like ResNet's skip connection.**
- *Lesson:* a gate held open is a shortcut for memory across time, like ResNet's `y = F(x) + x` across depth. Both give the gradient a path to survive.
- **Q (harder): "How do gates help with vanishing gradients?"**
  A: "A gate can pass the memory forward almost unchanged, like a skip connection, so the gradient has a short path back instead of being multiplied down at every step."

**7. Don't let information leak across the split.**
- *Lesson:* for time series, never shuffle before splitting, or the future leaks into training and the test score "is a lie".
- *Your project:* you are not forecasting, so a random split is allowed. But your clips were recorded back-to-back by the same person, so near-identical clips can end up in both train and test. That likely makes your test score optimistic, and you say so.
- **Q: "Could there be data leakage?"**
  A: "No clip is in two sets, and the test set was never used for any decision. But clips were recorded one after another by one person, so test clips look very similar to training clips. That's why I treat 100% as optimistic and list one signer as my main limitation."

### Session 08 — Attention (your Transformer)

**8. Attention: look at everything, weight what matters.**
- *Lesson:* instead of squeezing everything into one memory vector, keep every position and build a weighted blend. Steps: score, softmax, blend. The scaled dot-product version is `softmax(QKᵀ/√d)·V`.
- *Your project:* the Transformer uses `nn.TransformerEncoderLayer`, which does exactly this, with self-attention: every frame scores every other frame.
- **Q: "Explain attention simply."**
  A: "Each frame asks 'which other frames are relevant to me?' It scores all 90 frames, turns the scores into weights that add up to 1 with softmax, and takes a weighted mix. So frame 80 can look directly at frame 5 without passing through 75 steps of memory."

**9. Q, K, V and multi-head.**
- *Lesson:* Q = what I'm looking for, K = what each position offers, V = the content that gets mixed. √d keeps scores from getting too big. Multi-head runs several attentions in parallel, each learning a different relationship.
- *Your project:* 4 heads, model width 128 (`nhead=4, d_model=128`).
- **Q: "Why 4 heads?"**
  A: "Each head can focus on a different pattern, for example one on nearby frames and another on the start and end of the sign. 4 heads with width 128 is a small, standard choice for a small dataset; I didn't tune it."

**10. Where RNNs run out.**
- *Lesson:* RNNs must process step by step; attention connects every position in one step and can run in parallel.
- *Your project:* the Transformer has no recurrence. At your tiny size all three trained in 6–8 seconds, so you couldn't show a speed difference.
- **Q: "So why didn't the Transformer win?"**
  A: "It came very close, 98% vs 100%, which is one test clip, and the difference isn't significant. It also has the most parameters (307,083) for only 230 clips, and its curves show slight overfitting. Transformers usually need more data to beat recurrent models."

**11. Two parts not from the lecture files (likely the next session).**
- *Positional encoding:* attention alone doesn't know frame order. You add a fixed pattern of sines and cosines to each frame's vector so the model can tell frame 1 from frame 50.
- *Mean pooling:* after the encoder you average the 90 frame vectors into one summary vector, which plays the role of the RNN's final hidden state.
- *Linear 258→128:* projects each frame's 258 numbers to 128, like an embedding layer does for words (Session 07/08 embeddings), except your input is numbers, not word IDs.
- **Q: "How does your Transformer know the order of frames?"**
  A: "Through positional encoding: I add a fixed sine/cosine pattern that's different for every position, so frames carry their position with them."

### Session 03 — Loss, optimisers, overfitting

**12. Cross-entropy with softmax.**
- *Lesson:* for multi-class, the output layer gives raw scores (logits), softmax turns them into probabilities, and the loss is −log(probability of the true class). Confident mistakes are punished hardest. In PyTorch, `nn.CrossEntropyLoss` already contains softmax, so you feed it logits; applying softmax twice is a classic bug.
- *Your project:* `train.py` uses `nn.CrossEntropyLoss()`. The models return raw scores (the last layer has no softmax). Softmax is applied only in `evaluate.py` and `predict.py` to show probabilities.
- *Debug tip from the lesson:* the starting loss should be about −log(1/classes). For 11 classes that's 2.40, and your first-epoch losses were 2.35–2.40, so the setup was correct.
- **Q: "Which loss did you use and why?"**
  A: "Cross-entropy, the standard for multi-class classification. It punishes being confidently wrong. My model outputs raw scores because PyTorch's cross-entropy applies softmax itself; I only apply softmax when I want to display probabilities."

**13. Adam and the learning rate.**
- *Lesson:* Adam = momentum + a personal step size per weight; it "just works" at 0.001. The learning rate is the most important hyperparameter: too small is slow, too large bounces around.
- *Your project:* Adam at 0.001. Your tuning shows both failure modes: for GRU, 0.0001 was too slow (82–86%); for the LSTM, 0.001 was too large (48%) and 0.0001 worked better (70%).
- **Q: "Why Adam? Why that learning rate?"**
  A: "Adam is the recommended default and 0.001 is its usual starting point. Then I tested 0.001, 0.0005 and 0.0001. GRU was best at 0.001; the LSTM needed a smaller rate."

**14. Learning rate schedule.**
- *Lesson:* take big steps far from the minimum and small steps near it.
- *Your project:* `ReduceLROnPlateau` halves the learning rate when validation loss stops improving for 10 epochs.

**15. Epochs, batches, iterations.**
- *Lesson:* a batch is a small group of examples per update; an epoch is one full pass over the data.
- *Your project:* batch size 16; 230 training clips → 15 updates per epoch; up to 200 epochs, but early stopping ended each run after 44–48 epochs.
- **Q: "What is an epoch / a batch?"**
  A: "A batch is 16 clips used for one weight update. An epoch is one pass through all 230 training clips, which is 15 batches."

**16. Overfitting vs underfitting, read from the curves.**
- *Lesson:* underfitting = bad on training and test; overfitting = great on training, worse on validation; watch the gap between the training and validation curves.
- *Your project, one model of each kind:*
  - LSTM: training and validation both ~40% → **underfitting**.
  - GRU: both curves reach 100% together, no gap → **good fit**.
  - Transformer: training accuracy 100%, validation 94%; its validation loss bottomed at epoch 19 (0.13) and rose to 0.25 → **mild overfitting**.
- **Q: "How do you know the Transformer overfits?"**
  A: "In the loss curves, training loss keeps falling to about 0.01 while validation loss stops improving at epoch 19 and rises to 0.25. That growing gap is the overfitting sign from Session 03."
- **Q: "Why is the GRU's validation loss lower than its training loss?"**
  A: "Dropout is on during training and off during validation, so the training numbers are measured on a handicapped network."

### Session 04 — Regularisation and initialisation

**17. Dropout.**
- *Lesson:* randomly switch off units during training so no unit relies on another (the "football team" analogy); it acts like a cheap ensemble. Must call `model.eval()` at test time, the course's "#1 bug".
- *Your project:* dropout 0.2 after every layer in all three models. `evaluate.py` and `predict.py` call `model.eval()`.
- **Q: "What does dropout do and why 0.2?"**
  A: "During training it randomly turns off 20% of the units, so the network can't depend on any one unit, which reduces overfitting. At test time it's off; that's what `model.eval()` does. 0.2 is a mild, common value; I didn't tune it."

**18. Weight decay (L2).**
- *Lesson:* a constant "tax" that shrinks every weight; weights only stay big if the data keeps needing them. `weight_decay=1e-4` in the optimiser.
- *Your project:* your tuning tested weight decay 0 vs 0.0001 (the "one regularisation choice" the assignment requires). For GRU it made no difference; for the LSTM it helped a little (66% → 70% at the lowest learning rate).
- **Q: "Did regularisation help?"**
  A: "Weight decay made no difference for GRU, which already wasn't overfitting. For the LSTM it helped slightly. The Transformer, which does overfit, is where more regularisation would help next."

**19. Early stopping, keep the best weights.**
- *Lesson:* watch validation, keep a copy of the best weights, stop after `patience` epochs without improvement, and return the saved best weights, not the last ones.
- *Your project:* patience 25 on validation accuracy. Every time validation accuracy improves, the model is saved with `torch.save`, and that saved file is what gets tested. Best epochs: LSTM 21, GRU 23, Transformer 19.
- **Q: "Which epoch's model did you test?"**
  A: "The one with the best validation accuracy, not the last one. Training stops 25 epochs after the last improvement."

**20. "Perfect training performance is a warning sign."**
- *Lesson:* Student A memorised the exam and scored 100%, then failed the new one.
- *Your project:* your GRU scored 100%, so you checked whether it was too good to be true. The diagnostics found the dataset is easy (a no-order baseline also gets 100%). Use this as your example of acting on the lesson.

**21. What you did *not* use from Session 04, and should say honestly.**
- Explicit weight initialisation (Xavier/He): you used PyTorch's defaults.
- Data augmentation: none. For signs, a horizontal flip would need swapping the left and right hand data, because left and right can change the meaning; that's a future-work item.
- **Q: "Did you use data augmentation?"**
  A: "No. It's future work. For sign language I'd have to be careful: flipping the image swaps left and right hands, so I'd also need to swap the hand data so the label stays correct."

### Session 01 and 02 — Foundations

**22. Neurons, layers, activations.**
- Your classifier head is `Dense 64 → ReLU → Dense 11`: a small MLP on top of the sequence model. ReLU is the activation; inside LSTM/GRU, the gates use sigmoid (0 to 1) and the memory uses tanh (−1 to 1), as in Session 01's activation slides.
- **Q: "Where are activation functions in your model?"**
  A: "ReLU in the classifier head and in the Transformer's feed-forward layer; sigmoid and tanh inside the LSTM and GRU gates; softmax only to turn the final scores into probabilities."

**23. The training loop (forward, loss, backward, update).**
- *Lesson (Session 02):* initialise → forward → loss → backpropagate → update, repeat.
- *Your project:* the loop in `train.py` (and the same loop in the Colab notebook) is exactly these lines: `optimizer.zero_grad()`, `logits = model(xb)`, `loss = criterion(logits, yb)`, `loss.backward()`, `optimizer.step()`.
- **Q: "Walk me through your training loop."** See Part 3, `train.py`.

**24. scikit-learn vs PyTorch.**
- *Lesson:* sklearn hides the loop in `.fit()`; PyTorch gives full control and GPU support.
- *Your project:* models and training in PyTorch (required). scikit-learn is only used for the split, the metrics, and the two logistic-regression baselines in the diagnostics.

**25. "Deep learning needs lots of data."**
- Session 01 lists this as a con. With 30 clips per sign, data size is your main limitation and the reason the smaller GRU won.

### Session 05 and 06 — Indirect links (ideas, not code)

**26. Inductive bias / weight sharing (Session 05).** Session 05 calls convolution a "hard prior": built-in assumptions (locality, weight sharing). Your models have their own built-in assumptions: recurrence assumes the clip should be read in order and summarised in a memory; attention assumes any frame may relate to any other. That's what "inductive bias" means in your discussion slide.
- **Q: "What is inductive bias?"**
  A: "The assumptions a model makes before seeing data. An RNN assumes order matters and reads step by step; attention assumes any frame can relate to any other. Built-in assumptions help most when data is small."

**27. Counting parameters (Session 05).** You report trainable parameters for each model, as the lesson does when comparing architectures.

**28. Transfer learning (Sessions 05–06).** You did not use a pretrained model for the three approaches; all are trained from scratch. MediaPipe is pretrained, but it's a fixed preprocessing step (the "freeze everything, use the features" end of Session 06's dial), shared by all three models and not part of the comparison.
- **Q: "Is MediaPipe one of your models? Did you train it?"**
  A: "No. MediaPipe is a pretrained tool from Google that I use as a fixed feature extractor, the same for all three models. Every model I compare was trained from scratch by me."
- **Q: "Why didn't you use transfer learning?"**
  A: "My inputs are landmark numbers, not images, so there's no ImageNet-style pretrained model to reuse directly. A pretrained sequence model would be a good fourth approach in the future."

**29. Not used:** convolution, pooling, CNN architectures, detection, segmentation (Sessions 05–06), and encoder–decoder, teacher forcing, beam search, BLEU (Session 08).

### Session 14 — GANs

No overlap. If asked: "A conditional GAN could, in theory, generate extra synthetic sign clips to enlarge the dataset, but I didn't try it."

---

## Part 3 — Your code in plain language

How the files connect:

```
collect_data.py  → records clips (webcam + MediaPipe)        → data/<word>/<n>/landmarks.npy
extract_landmarks.py → stacks all clips into one dataset      → data/X.npy, y.npy, labels.npy
preprocess.py    → fixed 70/15/15 split (seed 42)            → data/X_train.npy … y_test.npy
train.py (and the Colab notebook) → trains one model         → models/<model>_khmer_sign.pt, results/<model>/history.json
evaluate.py      → tests it on the 50 test clips             → results/<model>/metrics.json, errors.csv, confusion matrix
compare.py       → one table and charts for all models       → results/comparison_table.md
diagnostics.py   → shuffle test, baselines, confidence ranges → results/diagnostics.json
predict.py / app/app.py → live webcam recognition
```

Shared helpers: `landmarks.py` (how a frame becomes 258 numbers), `dataset.py` (feeds arrays to PyTorch), `utils.py` (seed, device, saving checkpoints), `models/` (the three networks).

You actually trained on Colab with `notebooks/train_colab.ipynb`, which contains the same loop as `train.py`. If asked "where is your training code?", show either.

### `collect_data.py` — recording the dataset
- Opens the webcam, waits for SPACE, shows a 4-second countdown, then records 90 frames for each clip, 30 clips per word.
- Each frame is mirrored (selfie view), sent to MediaPipe, and turned into 258 numbers.
- The label comes from the word being recorded: everything is saved in a folder named after the word, so there is no manual labelling.
- **Q: "How did you label your data?"** "Automatically. The script records one word at a time and saves each clip in that word's folder."
- **Q: "Why is the image mirrored?"** "So it behaves like a mirror while I sign. The live predictor mirrors too, so training and use match. One side effect: MediaPipe's 'left hand' is my real right hand."

### `landmarks.py` — what one frame becomes
- Body pose: 33 points × (x, y, z, visibility) = 132 numbers. Each hand: 21 points × (x, y, z) = 63. Total 258.
- If MediaPipe can't find a hand in a frame, that hand's 63 numbers are set to zero.
- `SEQUENCE_LENGTH = 90` lives here, so recording, training and live prediction all agree.
- **Q: "Why landmarks instead of raw video?"** "258 numbers per frame instead of thousands of pixels. It's much smaller, it ignores background and clothing, and it trains in seconds."
- **Q: "What happens when a hand isn't detected?"** "It's stored as zeros. That's a limitation: the model might learn 'hand missing' as a clue, especially for sorry and thank you, which had the most missing frames (79% and 69% detected)."

### `extract_landmarks.py` — building the dataset
- Reads every clip, checks each is exactly 90 × 258 (skips any that aren't), and stacks them into `X.npy` (330 × 90 × 258), with the labels in `y.npy` and the word names in `labels.npy`.

### `preprocess.py` — the fixed split
- Splits once: 15% test first, then 15% validation, the rest training: 230 / 50 / 50.
- **Stratified**: every word has the same share in each set (about 21 / 4–5 / 4–5 clips per word).
- **Seed 42**, so the split is identical every time, and every model uses the same split.
- **Q: "Why train, validation AND test?"** "Training data fits the weights. Validation chooses the best epoch and the hyperparameters. The test set is touched only once at the end, so the final score is honest."
- **Q: "Why stratified?"** "So each word appears in each set in the same proportion; otherwise a word could be missing from the test set."
- **Q: "What does seed 42 mean?"** "It fixes the random numbers, so the split, the weight initialisation and the shuffling are the same every run, and anyone can reproduce my results."

### `dataset.py` — feeding data to PyTorch
- A small class that turns the NumPy arrays into PyTorch tensors and hands out one (clip, label) pair at a time. `DataLoader` then groups them into batches of 16 and shuffles the training set each epoch.

### `models/` — the three networks (see Part 2 for the theory)
- **`lstm_model.py`**: three LSTM layers (64 → 128 → 64 units) with dropout 0.2 between them. The memory after the last frame (`h_n[-1]`) goes to `Dense 64 + ReLU → Dense 11`. Output: 11 raw scores, one per word.
- **`gru_model.py`**: identical layout with GRU layers. Only the cell type changes, which makes it a fair comparison.
- **`transformer_model.py`**: `Linear 258→128` → add positional encoding → 2 encoder layers (4-head self-attention + feed-forward 256) → average over the 90 frames → `Dense 64 + ReLU → Dense 11`.
- `models/__init__.py` has `build_model("gru", …)`, so the same training script can build any of the three by name.
- **Q: "Why do the models output raw scores, not probabilities?"** "Because cross-entropy in PyTorch applies softmax itself. Adding softmax in the model would apply it twice."
- **Q: "Why the same layout for LSTM and GRU?"** "So the only difference is the cell type. Then any change in results comes from LSTM vs GRU, not from size."

### `train.py` — training one model (the Colab notebook does the same)
In plain words, per epoch:
1. For each batch of 16 training clips: clear old gradients (`zero_grad`), predict (`model(xb)`), measure the error (`criterion`), work out how each weight caused the error (`backward`), adjust the weights (`step`).
2. Check accuracy and loss on the 50 validation clips with the model in evaluation mode (no dropout, no learning).
3. If validation accuracy is the best so far, save the model (`torch.save`).
4. If the validation loss hasn't improved for 10 epochs, halve the learning rate.
5. If validation accuracy hasn't improved for 25 epochs, stop.

At the end it saves the curves (`history.json`) and facts about the run (`meta.json`: parameter count, training time, GPU).
- **Q: "What does `loss.backward()` do?"** "Backpropagation: it calculates, for every weight, how much it contributed to the error. Then `optimizer.step()` nudges each weight to reduce the error."
- **Q: "Why do you call `zero_grad()`?"** "PyTorch adds gradients up by default, so I clear them before each batch; otherwise old batches would affect the new update."
- **Q: "Why save checkpoints?"** "Colab can disconnect, and I want to test the best epoch, not the last one. The assignment also asks for `torch.save` checkpointing."

### Hyperparameter search (Colab cell / `hyperparam_search.py`)
- Trains the model 6 times, 60 epochs each, for every combination of learning rate {0.001, 0.0005, 0.0001} × weight decay {0, 0.0001}, with the same seed. Ranks them by **validation** accuracy (ties broken by validation loss), never the test set.
- **Q: "Why not choose settings by test accuracy?"** "Then the test set would be part of training decisions and the final score would be optimistic. Validation is for choosing; test is for reporting."

### `evaluate.py` — testing
- Loads the saved best model, switches to evaluation mode, predicts the 50 test clips, and computes accuracy, precision, recall, F1 and the confusion matrix. Saves every mistake to `errors.csv` with the confidence.
- **Q: "What are precision, recall and F1?"**
  "Recall: of all the real 'understand' clips, how many did I catch. Precision: of everything I *called* 'understand', how many were right. F1 combines both. Macro means averaged over the 11 words equally. Example: the LSTM had 100% recall but only 14% precision on 'understand', because it said 'understand' for almost everything."
- **Q: "What does the confusion matrix show?"** "Rows are the true word, columns the predicted word. Everything on the diagonal is correct. The Transformer's only off-diagonal cell is one 'understand' predicted as 'don't understand'."

### `compare.py` — one table for all models
- Reads every model's `metrics.json` and `meta.json` and builds the single results table and comparison charts that the assignment requires.

### `diagnostics.py` — the sanity checks
- **Frame-shuffle test:** shuffle the 90 frames of each test clip and test again. If accuracy barely drops, the model isn't relying on order. GRU: 100% → 94%.
- **No-order baseline:** average each clip over time (losing all order) and fit a simple logistic regression. It scores 100%, which shows the dataset is easy.
- **Confidence interval:** with only 50 test clips, one clip = 2 points. The 95% range for GRU is 93–100% and for the Transformer 90–100%; they overlap, so you can't say GRU is truly better than the Transformer.
- **Q: "Why did you do these extra checks?"** "My GRU got 100%, and Session 04 says a perfect score is a warning sign. I wanted to know whether the models really learned movement. They mostly didn't need to, because the signs already differ in hand shape and position."
- **Q: "Logistic regression isn't deep learning, why is it there?"** "It's only a baseline to measure how hard the data is; the assignment allows classical models as extra baselines. It's not one of my three approaches."

### `predict.py` and `app/app.py` — live use
- Keep the last 90 frames from the webcam in a rolling buffer. Once it's full, run the model each frame, and show the word if the model is at least 70% confident.
- **Q: "Does it work live?"** If you ran the live test, give your tally. If not: "Yes, it runs, but I haven't measured live accuracy; that's in my limitations. Live is harder because there's no countdown and the window slides over the start and end of the movement."

### `utils.py`
- `set_seed` (fixes all random numbers), `get_device` (GPU if available), `save_checkpoint` / `load_checkpoint`, `count_trainable_params`.

### Files that only make slides
`make_slide_figures.py`, `make_slide_art.py`, `make_deck.py` only draw charts and build the presentation from the saved results. They don't affect training.

---

## Part 4 — Rapid-fire questions

| Question | Short answer |
|---|---|
| What's the input and output? | 90 frames × 258 landmark numbers → 1 of 11 words. |
| How big is the dataset? | 330 clips, 30 per word, one signer; 230 / 50 / 50 split. |
| Which model won? | GRU 100%, Transformer 98% (one clip difference, not significant), LSTM 38%. |
| Why are the three approaches different enough? | Different architectures (assignment dimension A): gated recurrence with 3 gates, with 2 gates, and attention without recurrence. |
| Were they compared fairly? | Same data, same split, same seed, same optimiser and training rules, same test set and metrics. |
| Is 100% believable? | It's real on this test set, but optimistic: one signer, similar clips, and a no-order baseline also gets 100%. |
| Did you tune hyperparameters? | Learning rate × weight decay, 6 combinations, chosen on validation. GRU best at 0.001; LSTM needed 0.0001. |
| What would you do with more time? | Second signer; signs that differ only in movement; augmentation with left/right hand swap; body-relative coordinates; retrain the tuned LSTM; measure live accuracy. |
| How long did training take? | 6–8 seconds per model on a Colab Tesla T4. |
| Biggest limitation? | One signer in one room, so there's no evidence it works for other people. |
| How did you use AI? | Be honest and match your README note: what you used it for, and that you checked and can explain the code. |

### Numbers to remember

| | LSTM | GRU | Transformer |
|---|---|---|---|
| Test accuracy | 38% | 100% | 98% |
| Test errors (of 50) | 31 | 0 | 1 |
| Parameters | 236,811 | 178,827 | 307,083 |
| Training time (T4) | 6.7 s | 5.8 s | 8.1 s |
| Best epoch / epochs run | 21 / 46 | 23 / 48 | 19 / 44 |
| Frames shuffled | 35% | 94% | 93% |
| 95% interval | 26–52% | 93–100% | 90–100% |

Other facts: 258 = 132 + 63 + 63 · 90 frames ≈ 3 s · batch 16 (15 updates per epoch) · dropout 0.2 · Adam 0.001 · early stopping patience 25 · seed 42 · starting loss ≈ 2.40 = −log(1/11) · LSTM training accuracy ≈ 40% · LSTM tuned validation 48% → 70% · Transformer validation loss 0.13 (epoch 19) → 0.25 · 24 of the LSTM's 31 mistakes were "understand", all from one-handed signs · hands detected in 79% (sorry) and 69% (thank you) of frames.

---

## Practice

Before the presentation, open each file in `src/` and point to the line for each answer above: `h_n[-1]` in `lstm_model.py`, `nn.CrossEntropyLoss` and the five-step loop in `train.py`, `stratify=y` and `random_state=SEED` in `preprocess.py`, `model.eval()` in `evaluate.py`. The lecturer may ask you to show a line or change something small, for example "change the dropout to 0.3" (`dropout: float = 0.2` in each model file) or "use 60 clips per word" (`NUM_SEQUENCES = 30` in `collect_data.py`).
