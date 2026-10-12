# Slide content — copy and paste

How to read this file
- **ON SLIDE** = text to paste onto the slide (short on purpose).
- **VISUAL** = the picture to insert. All pictures are in `slides/figures/` and were generated from your real results. Re-make them any time with `venv\Scripts\python.exe src\make_slide_figures.py`.
- **SAY** = what to say out loud (not on the slide).
- **GET** = real data only you can collect, with how to get it. Anything in [square brackets] is a placeholder.

Structure follows Section 6.2 of the assignment. Timing target is 9:25 of the 10 minutes, so you have a buffer.
Style follows the example deck: footer "Student final project Deep Learning" and a page number on every slide, 16:9.
Colours used in the figures: GRU blue, Transformer green, LSTM orange. Use the same colours in tables and titles.

| # | Section (assignment 6.2) | Slide | Time |
|---|---|---|---|
| 1 | — | Title | 0:10 |
| 2 | Problem Definition & Motivation | Problem | 0:40 |
| 3 | Problem Definition & Motivation | System pipeline | 0:35 |
| 4 | Dataset & Pipeline | Dataset | 0:40 |
| 5 | Dataset & Pipeline | Split and preprocessing | 0:35 |
| 6 | Model Architectures & Strategies | Three approaches | 1:00 |
| 7 | Experimental Setup | Setup | 0:35 |
| 8 | Results & Visual Comparisons | Results table + chart | 0:50 |
| 9 | Results & Visual Comparisons | Learning curves | 0:45 |
| 10 | Results & Visual Comparisons | Hyperparameter tuning | 0:40 |
| 11 | Discussion & Error Analysis | Error analysis | 0:50 |
| 12 | Discussion & Error Analysis | Why GRU won, and what the results do not show | 0:55 |
| 13 | Discussion & Error Analysis | Limitations | 0:30 |
| 14 | Conclusion & Future Work | Conclusion and future work | 0:35 |
| 15 | — | Thank you / questions | 0:05 |

Backup slides A1–A5 are at the end. The example deck also had a table-of-contents slide; it is left out here to save time. Add it back (about 10 seconds) if you want to match it.

---

## Slide 1 — Title

**ON SLIDE**
```
Khmer Sign Language Word Recognition
Comparing LSTM, GRU and Transformer on MediaPipe landmark sequences

Presenter: [Your full name]
Course: Deep Learning
Lecturer: Mr. Soklong HIM
School: Kirirom Institute of Technology
Date: [presentation date]
Code and data: github.com/rinrin168/khmersignlanguage
```
**VISUAL** (optional): a screenshot of the MediaPipe skeleton on you while signing.
**GET**: run `venv\Scripts\python.exe src\mediapipe_test.py`, hold a sign, press Windows+Shift+S to capture the window, then press Q to close. It shows your face, so it is your choice whether to use it.
**SAY**: Name, one sentence: "I built a system that recognises 11 Khmer sign words and compared three neural networks on it."

---

## Slide 2 — Problem

**ON SLIDE**
```
Problem: recognising Khmer sign language words

• Task: classify a short webcam clip into one of 11 Khmer sign words
• Input: 90 frames × 258 body and hand landmark values (about 3 s)
• Output: one of 11 word classes (multi-class sequence classification)
• Why it matters: Khmer Sign Language has very little digital data
  and software compared with major sign languages
• Question: which sequence model recognises the signs best —
  LSTM, GRU or Transformer?
```
**VISUAL**: none needed.
**SAY**: Explain input and output precisely. Say the three models are the comparison; the rest of the talk answers the question on the last bullet.

---

## Slide 3 — System pipeline

**ON SLIDE**
```
From webcam to word

• MediaPipe Holistic (pretrained, used as a fixed feature extractor)
  turns each frame into 258 numbers
• 90 frames make one clip = one training example
• Only the sequence model is trained — from scratch, in PyTorch
• All three models get the same input and the same split
```
**VISUAL**: `slides/figures/fig_pipeline.png`
**SAY**: MediaPipe is not one of my compared approaches. It is a cited, fixed preprocessing step shared by all three. Everything that differs is the sequence model.

---

## Slide 4 — Dataset

**ON SLIDE**
```
Dataset: self-collected

• One signer (me), one webcam, one room
• 11 signs × 30 clips = 330 clips
• Each clip: 90 frames × 258 values (float32, 29 MB in total)
• Recorded with src/collect_data.py: 4 s countdown, then 3 s recording
• Labels come automatically from the sign being recorded
• Signs: how are you, again, correct, incorrect, don't understand,
  understand, deaf person, hearing person, delicious, sorry, thank you
```
**VISUAL**: `slides/figures/fig_dataset_split.png`
**SAY**: Self-collected, so I describe it honestly: one person, which is a limitation I come back to. Balanced: 30 clips per sign.
**GET** (so you can answer questions about the data):
- Recording conditions: how many sittings, lighting, background, which laptop camera. You know these; write one line, e.g. "[1 sitting, evening, room light, laptop webcam]".
- Real clip length. "About 3 s" assumes 30 fps. To measure it, run `venv\Scripts\python.exe src\mediapipe_test.py` and read "FPS" in the top-left box; 90 frames ÷ FPS = seconds per clip. If FPS is 15, the clip is about 6 s, so change the slide to match.
- License: it is your own data. Suggested wording: "Own recordings; shared in the repo for coursework."

---

## Slide 5 — Split and preprocessing

**ON SLIDE**
```
Split and preprocessing

• Fixed stratified split (seed 42): 230 train / 50 validation / 50 test
• Train: fit the weights
  Validation: pick the best epoch and the hyper-parameters
  Test: used once, for the final scores
• Identical for every model: MediaPipe landmarks, no augmentation,
  no extra normalisation
• A hand that is not detected is stored as zeros
```
**VISUAL**: `slides/figures/fig_hand_detection.png` (title it "Data quality: how often a hand was found").
**SAY**: The test set is never used for any decision. The chart shows missing-hand frames: sorry and thank you are the weakest (79% and 69% of frames).

---

## Slide 6 — Three approaches

**ON SLIDE**
```
Three architectures, same task

• LSTM: 3 recurrent layers with gates and a memory cell (236,811 parameters)
• GRU: same layout, simpler gates (178,827 parameters)
• Transformer: no recurrence — self-attention over all 90 frames
  at once (307,083 parameters)
• All trained from scratch, so only the architecture differs
• Why these three: the standard sequence models from class,
  from gated recurrence to attention
```
**VISUAL**: `slides/figures/fig_architectures.png`
**SAY**: LSTM and GRU read the clip frame by frame and keep a memory. The Transformer lets every frame look at every other frame. GRU has about three-quarters of the LSTM's parameters because it has 3 weight blocks instead of 4.

---

## Slide 7 — Experimental setup

**ON SLIDE**
```
Experimental setup

• Metrics: accuracy (main), macro precision / recall / F1, confusion matrix
• Loss and optimizer: cross-entropy, Adam, learning rate 1e-3, batch size 16
• Training: up to 200 epochs, early stopping (patience 25) on validation
  accuracy, learning rate halved when validation loss stalls
• Regularisation: dropout 0.2, weight decay 0 (tuned later)
• Tuning: learning rate {1e-3, 5e-4, 1e-4} × weight decay {0, 1e-4}
• Reproducible: seed 42 (Python, NumPy, PyTorch), torch.save checkpoints
• Hardware: Google Colab Tesla T4 · 6–8 seconds per model
```
**VISUAL**: none (it is a text slide). Optional: small table.
**SAY**: One sentence per bullet. The best epoch (by validation accuracy) is what gets tested.
**GET**: PyTorch version. It is printed in the second code cell of your Colab notebook ("Torch version: …"). Add "PyTorch [x.y]" to the hardware bullet.

---

## Slide 8 — Results

**ON SLIDE**
```
Results on the same 50-clip test set

| Approach    | Accuracy | Macro F1 | Parameters | Train time | Epochs |
|-------------|----------|----------|------------|------------|--------|
| GRU         | 100%     | 1.00     | 178,827    | 5.8 s      | 48     |
| Transformer |  98%     | 0.98     | 307,083    | 8.1 s      | 44     |
| LSTM        |  38%     | 0.31     | 236,811    | 6.7 s      | 46     |

• GRU: 0 errors out of 50 · Transformer: 1 · LSTM: 31
• 95% confidence intervals: GRU 93–100%, Transformer 90–100%,
  LSTM 26–52%
• GRU and Transformer cannot be separated with only 50 test clips
```
**VISUAL**: `slides/figures/fig_results.png` (left: accuracy with confidence bars; right: size and cost). Make the table a real PowerPoint table.
**SAY**: Same test set, same metrics for all. Be clear that the gap between GRU and Transformer is one clip.

---

## Slide 9 — Learning curves

**ON SLIDE**
```
Training vs validation curves

• LSTM underfits: training accuracy stays near 40%
• GRU: training and validation rise together, no gap
• Transformer: training reaches 100%, but validation loss
  turns up after epoch 19 (0.13 → 0.25): mild overfitting
• Early stopping kept the best epoch of each model (21, 23, 19)
```
**VISUAL**: `slides/figures/fig_curves_train_val.png`
**SAY**: Top row accuracy, bottom row loss on a log scale. Point at the Transformer's bottom panel: the two lines separate, which is the overfitting signature from class.

---

## Slide 10 — Hyperparameter tuning

**ON SLIDE**
```
Hyper-parameter tuning (validation accuracy, 60 epochs each)

• 3 learning rates × 2 weight-decay values = 6 runs per model
• GRU: 100% for learning rate 1e-3 and 5e-4; 1e-4 is too slow (82–86%);
  weight decay makes no difference → kept 1e-3 and 0
• LSTM: lower learning rate helps a lot, 48% → 70%
  (1e-4 with weight decay 1e-4)
• The reported LSTM row uses the default 1e-3; retraining it with the
  tuned values is future work
```
**VISUAL**: `slides/figures/fig_tuning.png`
**SAY**: Choices were made on validation accuracy only. This also tells us the LSTM's weak result is partly a settings problem.

---

## Slide 11 — Error analysis

**ON SLIDE**
```
Error analysis

• GRU: 0 errors. Transformer: 1 — "understand" read as "don't understand"
  (confidence 0.95)
• LSTM: 31 errors. 24 times it answered "understand", always with
  confidence ≈ 0.18 (chance is 0.09)
• All 24 came from the five one-handed signs → it learned which hands
  are present, not the finer movement
• Diagnosis: underfitting (training accuracy ≈ 40%), not overfitting
```
**VISUAL**: left `results/transformer/confusion_matrix.png`, right `slides/figures/fig_lstm_errors.png`.
**SAY**: The LSTM got sorry, thank you and understand fully right, and collapsed the one-handed signs into a single answer. I do not claim why the Transformer confused that one pair; that needs watching the two clips.
**GET** (optional, strengthens this slide): watch test clip number 39 (the Transformer's mistake) next to a typical "don't understand" clip and say whether the signs look alike. If you want that clip pulled out for you, ask me.

---

## Slide 12 — Why GRU won, and what the results do not show

**ON SLIDE**
```
Discussion

• GRU > LSTM: small data (230 clips) and 90 steps. GRU has fewer
  parameters and gates, so it is easier to optimise (capacity vs data size)
• Transformer close behind: attention reaches any frame in one step;
  its larger size (307k) shows as mild overfitting → regularisation matters
• Caution: shuffling the frames drops GRU only from 100% to 94%, and a
  logistic regression on clip averages (no order) scores 100%
• So the dataset is easy: the scores do not prove the models learned movement
```
**VISUAL**: `slides/figures/fig_diagnostics.png`
**SAY**: This is the most important slide for honesty. Left: scrambling the frames barely hurts GRU and Transformer. Right: a model with no concept of time already gets 100%. A harder test would use signs that differ only in movement, or a second signer.
Course terms used here: capacity, data size, regularisation, inductive bias (recurrence and attention are different assumptions about order).

---

## Slide 13 — Limitations

**ON SLIDE**
```
Limitations

• One signer, one room: no evidence of working for other people or cameras
• 30 clips per sign and a 50-clip test set: about ±7–10 points of uncertainty;
  clips recorded back-to-back may be near-duplicates across train and test
• Raw coordinates (not relative to the body); missing hands become zeros —
  possible shortcuts
• No augmentation; LSTM not retrained after tuning
• Live webcam accuracy not measured [or: measured, see next slide]
```
**VISUAL**: none.
**SAY**: Read them straight; this is rewarded in the rubric ("limitations clearly identified").
**GET** (optional): a live test. Run `venv\Scripts\python.exe src\predict.py --model gru`, sign each of the 11 signs 5 times, hold each for 3 seconds, and tally correct/total in a table (sign | tries | correct). Replace the last bullet with your number, e.g. "Live test: [x] of 55 correct".

---

## Slide 14 — Conclusion and future work

**ON SLIDE**
```
Conclusion

• Built a full pipeline: record → MediaPipe → 3 PyTorch sequence models →
  fair comparison (same split, seed, metrics)
• GRU best (100%), Transformer 98%, LSTM 38% (70% validation after tuning)
• Honest finding: the dataset is easy and frame order is barely used

Next steps
1. Record a second signer and test generalisation
2. Add signs that differ only in movement
3. Augmentation (time warping, flips with left/right swap) and landmark
   normalisation to the shoulders
4. Retrain the LSTM with its tuned settings
5. Live accuracy test

Code, data and results: github.com/rinrin168/khmersignlanguage
```
**SAY**: Three takeaways, then future work in priority order.

---

## Slide 15 — Thank you

**ON SLIDE**
```
Thank you
Questions?
```

---

# Backup slides (for the 5-minute Q&A; not counted in the 10 minutes)

## A1 — Per-class results (Transformer and LSTM)
**ON SLIDE** (real numbers from `results/*/metrics.json`)
```
Transformer: every sign has recall 100% except understand (75%, 3 of 4);
             don't understand precision 83%.
LSTM recall:  sorry 100%, thank you 100%, understand 100%,
              again 75%, hearing person 75%,
              correct, deaf person, delicious, don't understand,
              how are you, incorrect: 0%
LSTM precision: understand only 14% (4 of 28 "understand" answers were right)
```
**VISUAL**: `results/lstm/confusion_matrix.png` and `results/transformer/confusion_matrix.png`.

## A2 — Full tuning results
**ON SLIDE**: paste the two tables from `results/gru/hparam_search.json` and `results/lstm/hparam_search.json` (learning rate, weight decay, best validation accuracy, minimum validation loss, seconds). Or reuse `slides/figures/fig_tuning.png`.

## A3 — Architecture details
```
LSTM / GRU: 3 recurrent layers (64 → 128 → 64), Dropout 0.2 after each,
            last hidden state → Dense 64 + ReLU → Dropout → Dense 11
Transformer: Linear 258→128, sinusoidal positional encoding,
             2 encoder layers (4 heads, feed-forward 256, dropout 0.2,
             residual + LayerNorm), average over frames,
             Dense 64 + ReLU → Dropout → Dense 11
Loss: nn.CrossEntropyLoss on raw scores (logits)
```

## A4 — Data quality
**VISUAL**: `slides/figures/fig_hand_detection.png` if you did not use it on slide 5.
**ON SLIDE**: "Body pose found in 100% of frames. One-handed signs: deaf person, delicious, don't understand, how are you, incorrect, understand. Camera is mirrored, so the 'left hand' channel is your physical right hand."

## A5 — AI assistance disclosure
```
Tools used: Claude (Anthropic)
Scope: drafting and debugging the PyTorch pipeline, figures and README
Verification: code run and checked by me; analysis and conclusions are my own
```
(Match this to the AI-use note in the README. Edit the last line so it is true for you.)

---

# Questions you may be asked — short answers based on your real results

1. **Why these three models?** They are the standard sequence models from class: gated recurrence (LSTM, GRU) and attention (Transformer). They differ in architecture only, same data and training setup.
2. **Why did the LSTM fail?** It underfit: training accuracy about 40%. A no-order baseline gets 100%, so the task is not hard. Lowering the learning rate lifted validation accuracy from 48% to 70%, so settings are part of it. I did not test initialisation or sequence length.
3. **Is 100% believable?** It is consistent with an easy dataset: one signer, and a classifier on clip averages also scores 100%. With 50 test clips the 95% interval for 100% is 93–100%. It does not show the model works for other people.
4. **Do your models use the order of frames?** Only a little. Shuffling frames costs GRU 6 points (100 → 94) and Transformer 5 (98 → 93).
5. **Is MediaPipe one of your models?** No. It is a fixed pretrained preprocessing step, cited, used identically for all three. Every compared model is trained from scratch by me.
6. **How do you avoid leakage?** Fixed stratified split, test set used once, tuning and early stopping on validation only. Caveat: clips were recorded back-to-back, so near-duplicates can sit on both sides of the split.
7. **Why cross-entropy, and why does the model output raw scores?** Cross-entropy is −log of the probability of the true class, which punishes confident mistakes. `nn.CrossEntropyLoss` applies softmax internally, so the model returns logits; applying softmax twice is a classic bug. In the first epoch the training loss was 2.35–2.40 for all three models, which matches −log(1/11) = 2.40, the loss of pure guessing over 11 balanced classes.
8. **Why Adam with 1e-3?** A standard default that works without much tuning; the tuning search then tested lower values.
9. **What do dropout, weight decay and early stopping do?** All trade a little training accuracy for better generalisation. Dropout randomly turns off units; weight decay shrinks weights; early stopping keeps the epoch with the best validation accuracy.
10. **Why does GRU have fewer parameters than LSTM?** 3 weight blocks per layer instead of 4: 178,827 vs 236,811 (about three-quarters).
11. **How does attention work?** Each frame makes a query, key and value; attention weights are softmax(QKᵀ/√d) and each output is a weighted mix of the values. Multi-head does this several times in parallel. (Session 08.)
12. **Explain your training loop.** For each batch: `optimizer.zero_grad()`, forward pass, loss, `loss.backward()`, `optimizer.step()`; after each epoch evaluate on validation with `model.eval()`, keep the best checkpoint, stop after 25 epochs without improvement.
13. **What would you do next?** Second signer first, then movement-only signs, then augmentation and normalisation.

---

# Real data checklist (what only you can get)

| Item | Needed for | How to get it |
|---|---|---|
| Your name, presentation date | Slide 1 | You |
| Recording conditions (sittings, lighting, camera) | Slide 4, Q&A | You know; one line is enough |
| Real seconds per clip (FPS) | Slide 4 | Run `src\mediapipe_test.py`, read FPS, compute 90 ÷ FPS |
| PyTorch version | Slide 7 | Colab cell 2 output, or `venv\Scripts\pip.exe show torch` |
| Screenshot with MediaPipe skeleton | Slide 1 (optional) | `src\mediapipe_test.py` + Windows+Shift+S |
| Live test tally | Slide 13 (optional) | `src\predict.py --model gru`, 5 tries per sign |
| Watching the Transformer's one mistake | Slide 11 (optional) | Ask me to pull test clip 39 and a matching clip |
| Khmer script for each sign | README table | Lecturer or a Khmer Sign Language reference |
| A second signer's test | Slide 13/14 (optional) | Needs a small change to `collect_data.py` first (it currently overwrites the same `data/` folders), ask me |
