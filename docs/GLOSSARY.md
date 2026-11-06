# Glossary

Plain-language definitions for every technical term in this challenge, with football analogies
where they help. Grouped by theme; read top to bottom the first time.

---

## 1. The basic building blocks

**Frame**
One still image of the video. The video runs at **25 frames per second (fps)**, so 25 frames = 1
second, 1,500 frames = 1 minute. The match is 199,319 frames long. Our whole job is to put one label
on each frame.

**Label**
The answer we give for one frame: `H` (Barça have it), `A` (Athletic have it), `D` (ball dead) or
`X` (we decline to answer).

**Graded frames**
The frames that are actually scored: the two halves (158,857 frames). Pre-match, half-time and
post-match are ignored.

**Ground truth (GT)**
The "correct answers" we are scored against — the referee's notebook, if you like. Here it was
derived from *corrected* (human-cleaned) tracking. We only see the first 5 minutes of it; the rest
is hidden.

**Possession / spell / segment**
A **segment** is any unbroken run of the same label. A **possession** (the scorer calls it a
**spell**) is a segment of `H` or of `A`. So `H H H D D H H A A` contains two Barça possessions,
one Athletic possession and one stoppage. Note `H H D D H H` is **two** possessions — a throw-in
in the middle ends the first one, even if Barça take it.

**Transition / hand-over**
The moment possession switches team — a tackle won, an interception, a misplaced pass collected.
These are the hardest and most important frames.

**Dead ball**
Ball not in play: out for a throw-in/corner/goal kick, a foul, an offside, a goal celebration,
before a kickoff. Label `D`. We never say who "owns" a dead ball.

---

## 2. How you are scored

To understand the two metrics you need three ideas first.

**Precision**
"Of the things I *claimed*, how many were right?" If a striker takes 10 shots and scores 3, his
precision is 30%. A system that claims few things but is always right has high precision.

**Recall**
"Of the things that *really happened*, how many did I catch?" If a keeper faces 10 shots on target
and saves 7, his recall is 70%. A system that claims everything catches everything (high recall)
but is wrong a lot (low precision).

**F1 score**
A single number combining precision and recall (their *harmonic mean*):
`F1 = 2 × precision × recall / (precision + recall)`. It is only high when **both** are high —
you can't game it by being cautious or by being reckless. Ranges 0 to 1.

**Per-class F1**
F1 computed separately for each label. "F1 for `H`" asks: on how many frames did I say `H`
correctly, versus wrongly saying `H` and wrongly missing `H`?

**macroF1** *(the tie-breaker)*
The average of the three per-class F1s (`H`, `A`, `D`), computed **frame by frame**. "Macro" means
each class gets equal weight regardless of how many frames it has, so you can't score well by
always saying the most common label. Floor is about 0.16, not 0.
- Think of it as: *"how many minutes of the match did I label correctly?"*
- Weakness: a possession that flickers `H H A H H` for one frame barely hurts it (1 bad frame out
  of hundreds) even though it invented two extra possessions.

**IoU (Intersection over Union), temporal**
How well two time intervals overlap. Take a real possession (frames 100–200) and my predicted one
(frames 120–220): they share 80 frames (intersection) out of 120 frames covered in total (union),
so IoU = 80/120 = 0.67. Identical intervals → 1.0; no overlap → 0.

**Threshold**
A cut-off. "Matched at IoU 0.5" means the overlap must be at least 0.5 to count as the same
possession.

**spellF1** *(the ranking metric — this is what wins)*
Scores **possessions as whole objects**, the way a coach counts them:
1. Cut both the real labels and my labels into possessions (runs of `H` or `A`).
2. Pair each predicted possession with at most one real possession of the **same team**, best
   overlap first (*one-to-one matching*). A pair counts as a hit if its IoU is above the threshold.
3. Precision = hits / my possessions. Recall = hits / real possessions. Take F1.
4. Repeat at thresholds 0.3, 0.4, 0.5, 0.6, 0.7 and average the five F1s.

Football reading: *"Did I find each real possession, with roughly the right start and end, without
inventing fake ones?"* Constant guess scores exactly 0; perfect scores 1.

Why a flicker is a disaster here: one wrong frame in the middle of a 20-second Barça possession
turns 1 possession into 3 (`H…H`, `A`, `H…H`). Neither Barça half overlaps the real one enough,
and the fake `A` is a false claim — you lose a hit *and* add misses.

**Flickering** ★
Our answer switching between teams for a few frames when nothing changed on the pitch. An Athletic
player jogs past the Barça player on the ball, is 10 cm closer for 3 frames, and "nearest player"
hands Athletic the ball: `H H H A A H H` turns 1 possession into 3. Cheap on macroF1, very
expensive on spellF1. The trick is telling a real brief win of the ball from a player jogging past.

**Fragmentation**
Breaking one real possession into many small predicted ones. The number-one enemy of spellF1.

**Abstention (`X`) and forward-fill**
Writing `X` = "I don't know". macroF1 counts it as wrong. spellF1 first **forward-fills** it —
replaces each `X` with your last real label — so `H H X X H` becomes `H H H H H`. Therefore:
- `X` *inside* a possession (ball hidden behind players) is free and safe.
- `X` *across* a hand-over drags the old team's label past the change — costly.

**Confusion matrix**
A table with the true answer on each row and what we said on each column. The diagonal is what we
got right; everything off it shows exactly which mistake we make (e.g. row "truly D", column "we
said H" = dead balls we called Barça possession).

**Support**
How many true examples of a class there are (public 5 min: H 3,454 frames, A 2,577, D 1,469).

**Kappa**
Accuracy corrected for lucky guessing. 0 = no better than chance, 1 = perfect.

**transF1 / latency**
transF1: F1 for catching the moments possession changes, within ±1 second. Latency: how many
frames early (−) or late (+) our changes land.

**Team accuracy**
When the ball is truly alive, how often we name the right team. Measures "who" without "when".

**Share error**
How far off our overall possession percentage is.

**Noise / noise band**
Score differences too small to mean anything, because they could flip on luck. For spellF1,
gaps under ~0.04 on the full match are noise.

**Bootstrap**
A way to measure that luck: resample the possessions randomly many times, rescore each time, and
see how much the score wobbles. If two systems' wobble ranges overlap, you can't say one is better.

---

## 3. The data you get

**Tracking data / ATD (Automatic Tracking Data)**
For every frame, the x/y position and speed of each detected player and the ball, produced by
computer vision (software watching the video). "Automatic" = nobody cleaned it, so it has errors.

**Normalised coordinates**
Positions are given from 0 to 1 instead of metres: x=0 is one goal line, x=1 the other. Multiply
by 105.3 (length) and 68 (width) to get metres.

**Track slot / tracklet**
The tracker's numbered "lanes" (`Track 1`…`Track 46`). A slot is *not* a player — it's "some
player I'm following right now", and it can switch to someone else. We do know which **team** each
slot belongs to.

**Identity switch**
When the tracker swaps two players' slots — e.g. two players cross and it mixes them up.

**NaN (Not a Number)**
The value for "missing". The ball is `NaN` on 41% of graded frames — usually when it's in the air,
hidden in a crowd, or moving fast.

**Homography**
A small 3×3 table of numbers per frame that converts between a point on the pitch and the pixel
where it appears in the video. Lets you draw the tracking on top of the video to check it.

**Reprojection error**
How badly a homography misses. Lower = more trustworthy.

**Smart Tagging**
Metrica's automatic labelling of tactical phases (Build Up, Progression, Set Pieces, Transitions…),
each with a start time, end time and usually a team. Useful hints, not answers — it's also
automatic and makes mistakes.

**Tactical vs broadcast video**
*Tactical*: a wide fixed camera showing the whole pitch — our reference timeline. *Broadcast*: the
TV feed, with cuts and replays, on a different timeline. Optional and harder to use.

---

## 4. ML and modelling words

**Algorithm / model**
A recipe that turns inputs (tracking) into outputs (labels). A *model* usually means a recipe with
numbers learned from data; an algorithm can also be hand-written rules.

**Rule-based / heuristic**
Hand-written logic based on understanding of the game, e.g. *"if the ball leaves the pitch lines,
it's dead"*. No learning involved. Often very strong in problems like this one.

**Feature**
A number computed from raw data that helps make the decision — e.g. "distance from ball to nearest
Barça player", "average speed of all 22 players", "is the ball outside the touchline?".

**Baseline**
The simplest reasonable solution, used as a yardstick. Here: give the ball to whoever is nearest to
it, then smooth. Score to beat: spellF1 **0.2433**.

**Smoothing / majority-vote window**
Fixing flicker by looking at neighbouring frames. A ±50-frame window means: for each frame, look 2
seconds either side and take whichever label appears most. Like judging who's "on top" over a
spell of play, not from one touch.

**Hyperparameter**
A setting *you* choose rather than the model learning it — e.g. the size of the smoothing window.

**Tuning**
Trying different hyperparameter values and keeping the one that scores best.

**Fitting / calibrating**
Close cousins of training and tuning: adjusting a model's numbers so its outputs line up with some
reference data. Per the organisers' clarification, none of these may use the ground truth or the
provided video.

**External data**
Data about *other* matches (open tracking/event datasets we have the licence to use). Under the
organisers' clarification, it is the **only** data we may train or tune on.

**Licence (CC-BY, MIT)**
The legal terms for using a dataset. CC-BY 4.0: use freely, but credit the source. MIT: use
freely, keep the copyright notice. No licence at all = unclear rights, so ask.

**Domain gap**
The difference between the data you tune on and the data you're tested on — like preseason
friendlies on perfect pitches before a season of muddy away games. Ours: clean external tracking
vs our messy automatic tracking.

**Data degradation**
Deliberately making clean data messy (e.g. deleting the ball in long blackouts) so it resembles
the data we'll face. Shrinks the domain gap.

**Overfitting**
Getting very good at the examples you checked against, and worse at everything else — like a
team that drills set-pieces against one opponent's exact setup and falls apart against anyone else.
Tuning on the public 5 minutes does exactly this, which is why it's banned.

**Training / validation / test data**
*Training*: examples a model learns from. *Validation*: examples used to choose between versions.
*Test*: the final exam, never looked at. Here the full match is the test; we may not hand-label it
for training or validation.

**Leave-one-match-out**
Tune on all matches but one, score on the one left out, rotate so every match gets a turn, then
average. Like judging a manager's system on matches they didn't prepare for. A form of
cross-validation.

**Checksum**
A fingerprint of a file. If our download's fingerprint matches the publisher's, the file arrived
complete and unchanged.

**Supervised vs unsupervised learning**
*Supervised*: learning from examples with correct answers attached. *Unsupervised*: finding
patterns without answers. We have no labelled training data for this match, so pure supervised
learning on it isn't allowed.

**Pseudo-labels**
Labels produced by an algorithm (not a human) and then used to train another model. Allowed.

**Gradient-boosted trees**
A model made of hundreds of small yes/no question trees ("is the nearest Athletic player within
1.5 m?"). Each new tree fixes the mistakes of the ones before; their votes add up to a
probability. Strong on tables of features, fast, and explainable.

**Probability**
How likely something is, from 0 to 1. The frame judge says e.g. "H 70%, A 20%, D 10%" instead of
one hard answer, so the referee layer can weigh how sure it is.

**Test bench**
Our scoreboard: 14 Bundesliga halves made messy like our match, scored by the official metric
code. Every idea is judged here.

**Learning curve**
The score plotted against how much training data the model had. Still rising = more data would
help. Flat = spend the effort elsewhere.

**Data artefact / shortcut**
A pattern caused by how a dataset was made, not by football. The Bundesliga data freezes the ball
during stoppages; a model that learns "frozen ball = dead" fails on real tracking.

**Permutation importance**
How much a clue matters: shuffle that one clue's values and see how far the score drops.

**Nested leave-one-match-out**
Choosing settings without peeking at the test match: for each match left out, the settings are
picked on the other six only, then applied to it. Keeps the reported score honest.

**Team shape (width, block)**
How spread out a team is (width = spread across the pitch) and where its players sit on average
(block). In possession a team stretches wide; out of possession it gets compact. Our top clue for
"which team" because it works even when the ball is missing.

**Over- / under-segmenting**
Predicting too many possessions (flicker) or too few (merging real short ones). spellF1 punishes
both.

**Jitter / measurement noise**
Small random wobble in tracked positions. Real automatic tracking places a player or the ball about
a metre off, so "a player within 1.5 m" is less reliable than in clean data.

**Stress test**
Deliberately making the test harder (here, adding jitter) to see whether a gain survives. If it
shrinks to nothing, it was an artefact of clean data.

**Ceiling**
The best score a part of the system could ever give, measured by plugging in the true answer for
that part (on external data only).

**Random seed**
A fixed starting number for anything random (like where the fake blackouts go), so the result is
identical every run.

**State model / HMM (Hidden Markov Model)**
A model that assumes the match is always in one hidden "state" (Barça ball, Athletic ball, dead)
and that states tend to *persist* — switching has a cost. Each frame gives noisy evidence; the model
finds the most believable sequence of states overall. Natural fit for "possessions are long and
switches are rare".

**Viterbi algorithm**
The standard way to find that most believable sequence of states in an HMM.

**Interpolation**
Filling a gap by drawing a line between the known points either side — e.g. ball seen at frame 100
and frame 110, so estimate frames 101–109 on the line between them.

**Library (numpy, pandas)**
Ready-made code we import instead of writing ourselves. numpy does fast maths on big arrays of
numbers; pandas works with tables, like a programmable spreadsheet.

**Median / percentile (p90)**
Sort all values; the median is the middle one, p90 the one 90% of the way up. "Gap median 10.7 s"
means half of all ball blackouts last longer than 10.7 seconds. Less fooled by extreme values than
an average.

**Cache**
A saved, quick-loading copy of a slow result. We parse the 196 MB tracking file once and keep it
in `data/cache/` (git-ignored).

**Virtual environment (`.venv`)**
A private box of Python libraries for this project only, so versions don't clash with anything else
on the laptop. Git ignores it.

**Deterministic / reproducible**
Running the same code on the same data gives exactly the same file every time. Required here — the
organisers may rerun your repo and the output must match byte for byte. (Watch out for randomness;
fix random seeds.)
