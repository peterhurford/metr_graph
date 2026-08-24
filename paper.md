# Distillation and Chinese AI Progress: What We Know and How We Know It

*Working paper, August 24, 2026. Numbers labeled "this repo" are computed live from the
Epoch Capabilities Index (ECI) and Frontier Data Centers data in this repository by
`visualize_projection.py`; the functions named in §6 are the audit trail.*

---

## Summary

Distillation — training a model on another model's outputs — is simultaneously the most
discussed and the least quantified channel of US→China AI capability transfer. After
reviewing the documented record, the technical literature, and this repo's own
decomposition of frontier progress, the honest conclusions are:

1. **The technique works, cheaply, and API access is enough.** Black-box distillation of a
   deployed model's capabilities into a student costs 3–6 orders of magnitude less than
   creating those capabilities, and reasoning-trace distillation (DeepSeek's openly
   published R1-Distill family) demonstrably transfers frontier-adjacent math/code
   capability into small models. This much is documented and replicated.
2. **That Chinese labs distilled from US closed models at industrial scale is alleged in
   detail but never independently verified.** The evidence base runs from behavioral
   tells (models introducing themselves as ChatGPT or Claude) through fingerprint studies
   to the 2026 provider filings (OpenAI's memo to Congress; Anthropic's report of >16M
   Claude exchanges through ~24,000 fraudulent accounts attributed to MiniMax, Moonshot,
   and DeepSeek). Every strong claim rests on the accusers' undisclosed telemetry; no
   log has been audited, no case adjudicated, and every headline accusation landed at a
   strategically loaded moment in the chip-export debate.
3. **Distillation is a follower's accelerant, not a leader's engine.** The scaling-law
   and imitation literature agree the student is bounded by its teacher; distillation
   buys earlier *parity*, never a lead. The measured US–China frontier gap — Epoch's ~7
   months sustained since 2023; ~4 months on this repo's current pull — has tracked, not
   collapsed, which is what the ceiling story predicts.
4. **Quantitatively, it is real but third-order.** This repo's decomposition — to my
   knowledge the only attempt to put a number on the channel — measures a distillation
   premium of **~2.2 ECI points/year**, available only to sub-frontier models, out of a
   Chinese frontier pace of ~14 ECI/yr. In the model's US-pause counterfactual,
   distillation accounts for **14%** of the gap China closes, and cutting the channel
   entirely delays China's crossing by **~6 months** — behind indigenous innovation (50%)
   and the diffusion of published methods (19%). Because the term decays as the gap
   closes and today's gap is only ~5 ECI points, cutting distillation *today* buys the
   US less than one month against China reaching the current US frontier.
5. **The central open question is attribution, and everyone admits it.** No published
   analysis — hawkish or skeptical — decomposes Chinese progress into distillation vs.
   open-weight diffusion vs. indigenous R&D from provenance data; the estimates here are
   econometric inference from capability-vs-compute residuals, with the caveats that
   implies (§6.5).

Throughout, claims carry one of four epistemic labels: **[DOC]** documented (primary
source), **[REP]** reported (journalism, unnamed sources), **[ALL]** alleged (an
interested party's claim, evidence undisclosed), **[INF]** inferred (fingerprint,
benchmark, or econometric analysis).

---

## 1. What "distillation" means, and why the word does double duty

Knowledge distillation began as a compression technique: Hinton, Vinyals & Dean (2015,
[arXiv:1503.02531](https://arxiv.org/abs/1503.02531)) trained a small student on a large
teacher's softened output probabilities. Four variants matter for the China question:

- **Logit (white-box) distillation** requires the teacher's weights or full token
  distributions — available only for open-weight teachers or inside one lab (US labs
  routinely distill their own large models into cheaper ones).
- **Sequence-level (black-box) distillation** — sample the teacher's outputs via API,
  supervised-fine-tune the student on them (Kim & Rush 2016,
  [arXiv:1606.07947](https://arxiv.org/abs/1606.07947)). This is what makes closed
  weights an incomplete defense: the terms of service, not the technology, are the
  barrier. The lineage runs Alpaca (2023, 52K GPT-3.5 outputs, <$600) → Vicuna → Orca.
- **Reasoning-trace (CoT) distillation** — SFT on long chains of thought. DeepSeek-R1's
  paper ([arXiv:2501.12948](https://arxiv.org/pdf/2501.12948)) is the flagship [DOC]
  result: 800K R1-generated samples fine-tuned into Qwen and Llama bases produced
  R1-Distill-Qwen-32B, which beat OpenAI's o1-mini on AIME, MATH-500, and GPQA with no
  RL at all.
- **On-policy / teacher-graded distillation** — the student generates, the teacher
  grades per-token. Thinking Machines Lab (Oct 2025,
  [blog](https://thinkingmachines.ai/blog/on-policy-distillation/)) reports ~50–100×
  compute efficiency over RL to reach the same policy in math-reasoning settings [DOC].
  Adjacent extraction modes in the CNAS taxonomy: using a frontier model to clean/filter
  training data, or as a reward model.

Two distinctions the policy debate repeatedly collapses:

**Distillation-as-technique vs. distillation-as-accusation.** DeepSeek *openly* distilled
R1 into six Qwen/Llama checkpoints and published the recipe [DOC] — legitimate, licensed,
and on Hugging Face. The accusation is something else: covertly harvesting a *rival's*
API outputs against its terms of service. Much 2025 commentary conflated the two; the
conflation is itself a finding. OpenAI's own Feb 2026 memo takes care to distinguish
"legitimate use cases for distillation" from "imitation frontier AI models."

**Distillation vs. the other two diffusion channels.** Capability flows from the US
frontier to followers through three distinguishable channels: (a) distillation from
closed models' outputs; (b) **open-weight releases** — R1-Distill's students sit on Llama
and Qwen *weights*, and Qwen is the most-forked ecosystem on Hugging Face; (c)
**published methods** — and this channel runs both ways: DeepSeek's MLA attention, GRPO,
and FP8 recipes went east-to-west and are now industry standard [DOC]. Any claim about
distillation's impact has to hold the other two channels fixed; §6 is built around
exactly that separation.

---

## 2. The documented record: what actually happened

### 2.1 The DeepSeek episode (December 2024 – 2025)

DeepSeek-V3 shipped Dec 26, 2024, claiming $5.576M of marginal pretraining compute
[DOC, self-reported]; R1 followed Jan 20, 2025, and in September 2025 became the first
frontier LLM through journal peer review (Nature, Vol. 645), with the RL stage priced at
~$294K [DOC]. Within a day of V3's release, TechCrunch documented it identifying itself
as ChatGPT in 5 of 8 trials and reproducing GPT-4 jokes verbatim [INF]
([link](https://techcrunch.com/2024/12/27/why-deepseeks-new-ai-model-thinks-its-chatgpt/)).

The accusation wave came the week R1 cratered Nvidia's stock:

- **Jan 28, 2025** — David Sacks (White House AI czar): "substantial evidence" DeepSeek
  "distilled the knowledge out of OpenAI's models." No evidence shown [ALL].
- **Jan 29** — Bloomberg [REP, unnamed sources]: Microsoft security researchers had
  observed, in fall 2024, individuals believed linked to DeepSeek exfiltrating large
  volumes of data via OpenAI's API; accounts were banned.
- **Jan 29** — OpenAI to the FT and Axios: "some evidence" of distillation "suspected to
  be from DeepSeek"; evidence withheld [ALL]. OpenAI never sued, and commentators noted
  it had little legal recourse beyond terms-of-service breach — while itself defending
  training-data suits.

DeepSeek's only on-the-record answer came through Nature's peer review (Sept 2025)
[DOC]: V3's crawled corpus contained "a significant number of OpenAI-model-generated
answers," but incidentally — "all data used in this phase were naturally occurring" —
and R1 "did not learn by copying reasoning examples generated by OpenAI models."

Fingerprint studies point both ways: Copyleaks (Mar 2025) classified 74.2% of R1 outputs
as matching OpenAI's stylistic fingerprint [INF, vendor with product interest]; by June
2025, independent researchers found R1-0528's lexical preferences resembling *Gemini*
instead, suggesting a switched teacher — or shifting contamination [INF, weak].

### 2.2 Precedents and the broader roster

- **ByteDance (Dec 2023)** is the earliest concrete case: The Verge reported internal
  documents showing OpenAI API use across Project Seed's development [REP]; OpenAI
  suspended the account [DOC]; ByteDance admitted "annotation" use only [DOC].
- **The distillation-quantification paper** ([arXiv:2501.12619](https://arxiv.org/abs/2501.12619),
  Jan 2025; ACL 2025) — notably by *Chinese* researchers (CAS Shenzhen, PKU, 01.AI) —
  ranked GLM-4-Plus, Qwen-Max, and DeepSeek-V3 high on "distillation degree" via
  identity-leakage and GPT-4o response similarity; Claude, Doubao, and Gemini scored
  low. Qwen-Max carried substantial *Claude*-derived identity content [INF].
- **Huawei Pangu (Jul 2025)**: fingerprinting analysis and a self-described internal
  whistleblower alleged Pangu was upcycled from Alibaba's Qwen [INF+ALL, denied].
  Distillation disputes — and their deniability problems — are intra-Chinese too.

### 2.3 The 2026 escalation: from anonymous sourcing to formal filings

The record's character changed in February 2026: the accusations became primary-source
documents, though the underlying telemetry stayed closed.

- **Feb 12, 2026 — OpenAI's memo to the House Select Committee**
  ([PDF](https://assets.bwbx.io/documents/users/iqjWHBFdfxIU/rRmql_jJcxb4/v0))
  [DOC-of-allegation]: "accounts associated with DeepSeek employees" developing
  circumvention methods, access "through obfuscated third-party routers," code "to
  access US AI models and obtain outputs for distillation in programmatic ways." Most
  adversarial distillation "appears to originate from China." No raw evidence appended;
  a closed-door briefing offered.
- **Feb 12, 2026 — Google GTIG** reported rising distillation attacks on Gemini,
  including one campaign of >100,000 prompts engineered to elicit full reasoning traces
  [DOC vendor report].
- **Feb 23, 2026 — Anthropic, "Detecting and preventing distillation attacks"**
  [DOC-of-allegation]: >16M Claude exchanges through ~24,000 fraudulent accounts and
  "hydra cluster" proxy architectures — **MiniMax >13M** exchanges, **Moonshot >3.4M**
  (agentic reasoning, tool use, coding, vision), **DeepSeek >150K**. A notable technical
  detail: DeepSeek prompts allegedly asked Claude to "imagine and articulate the internal
  reasoning behind a completed response" — synthesizing chain-of-thought training data at
  scale to route around hidden reasoning traces. None of the three firms responded.
- **Apr 23, 2026 — White House OSTP memorandum NSTM-4**, "Adversarial Distillation of
  American AI Models" [DOC government position]: "deliberate, industrial-scale
  campaigns"; directs intelligence-sharing with labs — but imposes no new export
  controls or entity listings, and calls legitimate distillation "vital."
- **Jul 2026 — the Kimi K3 flashpoint.** K3 (Jul 16, 2.8T parameters, billed as the
  largest open-weight model) produced viral "I'm Claude" screenshots [INF]; OSTP
  director Kratsios accused Moonshot of "large-scale, covert industrial distillation"
  using GB300 servers in Thailand [ALL; no penalties announced]. The strongest
  *independent* technical evidence in the entire record is the
  ["Which Claude is K3?"](https://github.com/rgreenblatt/which_claude_is_k3/blob/main/writeups/write_up.md)
  probe study [INF]: under prefill, K3 emitted exact Anthropic API version strings
  (`claude-opus-4-5-20251101`, 12 times) that real Claude models almost never state —
  hard to explain by generic web contamination, and pointing to labeled Claude
  transcripts with deployment metadata in K3's training data. Who collected them, and
  how, remains open. Named skeptics (Hancock, Lambert) note the timeline makes "K3
  distilled from Fable" implausible — Fable was public ~15 days before K3 shipped —
  though distillation from *earlier* Claude models is a separate question.

### 2.4 Countermeasures, and whether they worked

The levers actually deployed [DOC]: region blocks (OpenAI cut unsupported regions
including China, Jul 2024); hidden chains of thought (o1 onward, partly explicitly
anti-distillation); KYC ("Verified Organization" with government ID, OpenAI, Apr 2025);
ownership-based bans (Anthropic barring entities >50% Chinese-owned regardless of
operating location, Sep 2025, at a self-estimated cost in the hundreds of millions);
behavioral detection and account-cluster forensics (both labs' 2026 filings). No lab
announced deployed output watermarking against distillation, though the method exists
(watermark "radioactivity," [arXiv:2402.14904](https://arxiv.org/abs/2402.14904)) and so
do published attacks on it.

The best evidence the measures did **not stop** extraction is the accusers' own 2026
testimony: OpenAI describes activity as "evolving but persistent," changing "in part
because we have added new methods" — displacement to resellers and routers, not
cessation; Anthropic's hydra clusters (one proxy network managing >20,000 accounts) were
caught only by cross-account behavioral analysis [DOC-of-claim]. Both labs converge on
an ecosystem argument: hardening one provider pushes traffic to the least protected one.

### 2.5 Reading the record honestly

Three structural facts should discipline any conclusion drawn from §2:

1. **The strongest evidence is the least verifiable.** Provider telemetry — the only
   evidence that could establish scale and intent — has never been independently
   audited. Everything public is either weakly diagnostic or circumstantial.
2. **The universal confounder is contamination.** Post-2023 web text is saturated with
   GPT- and Claude-generated content, so "the model says it's ChatGPT" is consistent
   with either deliberate harvesting or passive crawling. Only artifacts implausible
   under contamination (K3's current API version strings; audited account-level logs)
   discriminate — and there is exactly one of the former in the public record.
3. **Every headline accusation was strategically timed** — the Jan 2025 wave against
   R1's market shock, the Feb 2026 filings amid the H200 export debate, the Anthropic
   post while Washington weighed relaxing chip controls. That does not falsify the
   claims; it belongs in the weighting.

---

## 3. What the science says: efficacy and limits

### 3.1 It works, and it is cheap

The cost ladder, each rung documented (§ sources in the table):

| Layer | Cost |
|---|---|
| Frontier US pretraining run | >$100M |
| DeepSeek-V3 base pretrain (marginal GPU cost, self-reported) | $5.6M |
| R1 RL stage (peer-reviewed, Nature) | $294K |
| R1→student CoT distillation (800K samples, SFT) | ~$10K–100K class [INF] |
| s1-class curated distillation (1K traces, [arXiv:2501.19393](https://arxiv.org/abs/2501.19393)) | ~$50 |

Once a teacher is *deployed*, replicating a targeted capability costs 3–6 orders of
magnitude less than creating it. (Counter-accounting: SemiAnalysis put DeepSeek's
company-level position at ~50K Hopper GPUs and ~$1.6B capex [REP] — headline per-run
costs understate the resources behind even the student labs.)

### 3.2 It is teacher-bounded

The "Distillation Scaling Laws" paper (Busbridge et al., Apple,
[arXiv:2502.08606](https://arxiv.org/abs/2502.08606), ICML 2025) [DOC] gives the
quantitative frame: student loss is predictable from the compute split; distillation
beats supervised training only below a compute threshold; a too-strong teacher *hurts*
the student (the capacity gap); and the student's achievable loss is governed by the
teacher's. Distillation is a machine for approaching a teacher cheaply, not for passing
it. The known exceptions (born-again self-distillation; weak-to-strong generalization)
are the wrong configuration for the China case, and pushing past a teacher requires new
signal — verifiable-reward RL — which is how the frontier itself moves and is not
distillation.

Equally load-bearing is the older negative result: Gudibande et al., "The False Promise
of Imitating Proprietary LLMs" ([arXiv:2305.15717](https://arxiv.org/abs/2305.15717))
[DOC] — imitation models copy style, not broad capability; outside the imitation data's
coverage they "close little to none of the gap," which is bridged by better *base
models*, i.e., pretraining compute. Small students learn badly from strong teachers
([arXiv:2502.12143](https://arxiv.org/abs/2502.12143)). And the RL flywheel itself is
not extractable by API — only its products, the traces, are (Lambert,
[Interconnects](https://www.interconnects.ai/p/how-much-does-distillation-really) [REP,
expert judgment]).

Epoch AI's assessment ("Keeping up with the GPTs," Apr 2026,
[link](https://epoch.ai/gradient-updates/keeping-up-with-the-gpts/)) is the most careful
third-party bound: distillation is the most compelling efficiency lever available to a
compute-poor lab, worth "several-fold" compute savings on specific benchmarks — and
efficiency levers together cannot close a 10× compute gap [DOC analysis].

### 3.3 The prediction this makes

If distillation were a frontier-erasing exploit, the US–China lag should compress toward
the API-availability delay (weeks). If it is a bounded accelerant, the lag should
persist while followers track the frontier from below. The data match the second:
Epoch measured a ~7-month average Chinese lag sustained since 2023 (range 4–14,
[data insight](https://epoch.ai/data-insights/us-vs-china-eci)) [DOC], "sustained rather
than closing differences in development velocity" — and noted the US–China gap closely
resembles the closed-vs-open-weights gap, since leading Chinese models are open-weight.
On this repo's current Epoch pull (Aug 2026) the frontier gap is 5.0 ECI points —
Claude Fable 5 (162.5, Jun 2026) vs. Kimi K3 (157.5, Jul 2026) — about **3.9 months** at
the US frontier pace, near the bottom of Epoch's historical range but not qualitatively
off it. Frontier slopes since 2024: US ~15.3 ECI/yr, China ~13.9.

---

## 4. Quantifying the impact: outside estimates

The striking fact is how little exists. As of this writing:

- **No published analysis attributes a number of months or a compute discount to
  distillation from US closed models.** CNAS's hawkish "Adversarial Distillation"
  report (Jun 2026) says so explicitly — isolating the contribution "would require
  replicating Chinese training runs with/without distilled data." CSET's
  Shea-Blymyer: "We are essentially guessing at the extent of the threat" [DOC
  admissions].
- The best partial quantifications are order-of-magnitude sanity checks: Lambert
  estimates the alleged Claude campaigns total 150–400B tokens — enough to
  "meaningfully improve post-training" but post-training-scale, not pretraining-scale
  (R1's own SFT set was ~6.4B tokens); Epoch's "several-fold savings on specific
  benchmarks" [INF, named analysts].
- Even the accusers frame distillation as an accelerant — "capabilities in a fraction
  of the time, and at a fraction of the cost" (Anthropic) — not a substitute for
  training. And the lab with the *least* alleged Claude contact (DeepSeek, >150K
  exchanges vs. MiniMax's >13M) is among China's strongest, "suggesting distillation
  wasn't determinative" (Lawfare, Jul 2026) [DOC analyses].

The skeptical case in full: indigenous Chinese algorithmic innovation is real and
peer-reviewed (MLA, GRPO, FP8, pure-RL reasoning); the open-weights channel (Llama,
Qwen) is a distinct and arguably larger diffusion path no API control touches; compute
remains the binding constraint by DeepSeek's own CEO's testimony ("The problem we face
has never been funding, but the export control on advanced chips"); and the lag data
show tracking, not convergence.

---

## 5. Quantifying the impact: this repo's decomposition

Against that near-vacuum, this repository's dashboard makes — with appropriate
humility — an actual attempt at the number. The approach is econometric: distillation is
identified not from provenance (who queried whose API) but from its *signature* in the
capability-vs-compute record — followers with external teachers improving faster at
fixed compute than the frontier can. All figures below are live from the Aug 2026 Epoch
pull; function names refer to `visualize_projection.py`.

### 5.1 Three measured signatures

**Signature 1 — the coefficient shift (`_cc_frontier_grade_algo`).** Regressing ECI on
log₁₀(training FLOP) and time over all 174 disclosed-compute models gives ~8.5 ECI per
×10 compute and ~12.8 ECI/yr at fixed compute. Refit on near-frontier models only —
those within 5 ECI of the running frontier at release, the subset least able to lean on
a stronger teacher — and *both* coefficients move: compute rises to ~10.3 (11.25 within
±3 ECI), time falls to ~10.9 (8.8). That two-way gradient is what distillation predicts:
teacher-fed followers get capability without compute, flattening the compute slope and
steepening the time slope in the pooled fit. It is the model population's fingerprint of
distillation, measured without any provenance claim.

**Signature 2 — the iso-compute premium (`_cc_iso_compute`).** Holding the compute
budget fixed and watching ECI climb: models in the 10^23.5 and 10^24.5 FLOP bands gain
**+13.3 and +13.9 ECI/yr**, while the top band (10^25.5) — the models nearest the
frontier, with the least external teaching available — gains only **+11.1**. The
difference, **~2.2 ECI/yr**, is the distillation premium: capability growth available
only to models with a stronger teacher to learn from.

**Signature 3 — the vanished country premium (`_cc_iso_compute_rate`).** China's
iso-compute algorithmic rate (12.55 ECI/yr, n=50) now sits within 1% of the US rate
(12.43, n=22). Earlier pulls showed a larger Chinese premium; as the gap narrowed to ~5
points, the premium shrank — which is exactly what a gap-decaying distillation term
predicts, though the US fit's small n makes this suggestive rather than conclusive.

### 5.2 The channel decomposition

The dashboard splits frontier ECI growth into four channels, each measured or bounded
independently (`_cc_pure_innovation_band`, `_cc_innovation_algo_band`):

| Channel | ECI/yr | Identification |
|---|---|---|
| Physical compute | exchange rate (~10.3/×10) × each side's capacity growth | data-center buildout |
| Innovation (never decays) | 4.5–8.8 | pretraining-efficiency prior × exchange rate, up to the ±3 near-frontier refit |
| Diffusion of published methods (dries ~1yr after a pause) | ~3.3 | no-external-teacher level minus innovation |
| **Distillation (decays as the gap closes)** | **~2.2** | all-band iso-compute minus the top band |

Distillation applies only to the follower (the frontier has no stronger teacher);
diffusion flows both ways. The consistency check: computing innovation as the *residual*
of each country's observed frontier slope lands inside the independently derived
4.5–8.8 band for both countries.

### 5.3 What the channel is worth in months

The projection engine (`_cc_cn_crossing_sim`) encodes the ceiling argument directly:
the distillation term scales with min(1, gap/gap₀) — you cannot overtake your teacher —
while diffusion decays only after publications stop and innovation never decays. Two
policy-relevant results:

**Today, with a 5-point gap, cutting distillation buys almost nothing.** China's median
crossing of ECI 161 (the current US frontier level) is Nov 2026; cutting the
distillation channel today moves it to Dec 2026 — **+0.6 months**. The term has mostly
already decayed.

**In the wide-gap counterfactual, it is worth months, not years.** The Pacing tab's
US-pause scenario (US halts at its first 10^28-FLOP run, frozen frontier ≈ ECI 191;
China closes ~33 ECI over ~31 months) decomposes the same simulated paths
(`_pc_render_why`):

| Channel | ECI closed | Share | Crossing delay if removed |
|---|---|---|---|
| Indigenous innovation | +16.5 | 50% | not by 2031 |
| Diffusion (published methods) | +6.3 | 19% | +8.9 mo |
| **Distillation** | **+4.6** | **14%** | **+6.3 mo** |
| Compute — remote access abroad | +3.2 | 10% | +4.4 mo |
| Compute — domestic clusters | +2.5 | 8% | +4.7 mo |

The "delay if removed" column is deliberately non-additive: kill one channel and the gap
stays wider, so the distillation term runs at full strength longer — the channels partly
cover for each other. Note the ordering: even in the scenario most favorable to
distillation mattering (a paused US frontier sitting still as a fixed teacher), the
channel ranks third, behind innovation and open publication.

### 5.4 Convergence with the outside estimates

Three independent lines land in the same place. The literature's qualitative bound
(post-training-scale accelerant, "several-fold on specific benchmarks, cannot close a
10× compute gap"); the measured lag (persistent ~4–7 months, not collapsing); and this
model's decomposition (~2.2 ECI/yr ≈ 15% of the Chinese frontier's pace; a ~6-month
crossing delay if severed at wide gap). Nobody's evidence supports either extreme —
"distillation built DeepSeek" or "distillation is irrelevant."

### 5.5 Caveats on the repo's numbers

These are honest limitations, not boilerplate. (1) The identification is *residual*, not
provenance: the "distillation" channel is whatever makes sub-frontier models improve
faster at fixed compute — which includes distilling from open-weight Chinese teachers
and internal self-distillation, not only illicit US-API harvesting. It is best read as
an upper bound on the US-closed-model channel specifically. (2) Compute and time are
collinear (r≈0.58), so the pooled split is approximate; the iso-compute construction
sidesteps but does not eliminate this. (3) The band fits are small-n (the top band has
27 models; the US country fit 22). (4) ECI bundles post-training, so this is
total-capability efficiency. (5) Epoch recomputes scores live; the numbers drift
between pulls. (6) The channel-decay laws (gap-proportional distillation, one-year
diffusion absorption) are modeling assumptions with the right qualitative shape, not
measured decay curves.

---

## 6. Policy reading

**Export controls and distillation are complements, not substitutes.** The
"distillation undermines chip controls" argument (CNAS, DSET's "distillation cascade,"
Just Security) is half right: model outputs are an uncontrolled channel, and the 2026
record shows API-level enforcement leaking badly through resellers and proxy clusters.
But the RAND/Epoch counter holds up quantitatively here: distillation's measurable yield
is ~2 ECI/yr against a compute channel whose exchange rate is ~10 ECI per ×10 FLOP, the
synthetic-data flywheel itself consumes compute, and DeepSeek's CEO named chips, not
knowledge, as the binding constraint. In this repo's pause decomposition the two
compute rows together (18%) outweigh distillation (14%) even before counting compute's
role in feeding the algorithmic channels.

**Enforcement buys months, and only at wide gaps.** The model's cleanest policy output:
a perfectly enforced distillation cut is worth ~6 months when the gap is tens of ECI
points (the pause scenario) and ~0.6 months at today's 5-point gap. The time to sever
the channel is *before* or *during* a divergence — e.g., at the start of any US
capability run-up or pause — not after parity is nearly reached. Meanwhile the 2026
record shows actual enforcement achieving displacement, not cessation [DOC-of-claim],
which discounts even the 6 months.

**The larger channels get less attention.** Published methods (19%) outrank distillation
(14%) in the decomposition, and the open-weights channel — through which US Llama
releases and Chinese Qwen releases both diffuse — is the confound Epoch flags in the lag
data itself. A policy conversation focused exclusively on API distillation is aimed at
the third-largest channel.

**Legal form matters.** Model outputs are not copyrightable; the viable theories are
fraud-of-access (CFAA-style), not IP theft; and quasi-IP in outputs would entrench
incumbents who themselves trained on scraped text (Lawfare). NSTM-4's restraint — no new
listings, legitimate distillation called "vital" — reflects this.

---

## 7. What we don't know, and what would change the picture

The genuinely open questions, in descending order of importance:

1. **The causal attribution.** No one has decomposed Chinese progress into
   distillation / open-weight diffusion / indigenous R&D from provenance data. This
   repo's residual decomposition is one indirect estimate; an audit of any accuser's
   telemetry, a with/without training-run replication, or deployed watermark
   radioactivity statistics would each be worth more than everything currently public.
2. **Scale and intent behind the 2026 allegations.** The 16M-exchange, 24K-account
   figures are precise-sounding but unaudited [ALL]. Independent verification — or a
   court case with discovery — would move them to [DOC] and would also calibrate how
   much extraction the countermeasures actually intercept.
3. **Whether the RL era degrades the channel.** If frontier capability increasingly
   lives in on-policy RL loops that cannot be extracted through outputs — the
   Lambert/Forbes claim — the distillation premium should shrink in future pulls of
   Signature 2. That is a testable prediction this dashboard will register.
4. **Whether the vanished country premium (Signature 3) stays vanished.** If China's
   iso-compute rate re-diverges upward while the gap re-widens, the gap-decay law gets
   direct support; if it diverges while the gap stays small, the model is wrong about
   the mechanism.
5. **The teacher-labeled-transcript question.** The K3 version-string result implies
   labeled Claude transcripts circulate as training data. Whether via in-house
   harvesting, third-party data brokers, or public transcript dumps matters enormously
   for enforcement design and is completely unknown.

---

## 8. Bottom line

Distillation from US models is real, cheap, documented as a technique, and credibly —
but never verifiably — alleged as an industrial practice of Chinese labs. Its ceiling is
its teacher; the measured record (a persistent multi-month lag, followers tracking from
below) and this repo's decomposition (~2 ECI/yr, ~14% of a catch-up, ~6 months if
severed at wide gap and under one month today) agree it is a second-or-third-order
accelerant of Chinese progress — behind indigenous innovation, behind published methods,
roughly on par with compute access. The loudest claims on both sides outrun the
evidence: nobody has shown distillation *drove* Chinese frontier progress, and nobody
has shown it negligible. What we know best is how much we don't: the central
quantitative question — attribution — has exactly one public estimate with an audit
trail, and it is the one in this repository.

---

*Sources are linked inline. External-evidence sections synthesize primary documents
(the OpenAI Feb 2026 memo, Anthropic's Feb 2026 report, NSTM-4, the House Select
Committee report, the R1/V3/scaling-law papers) and dated secondary reporting; each
claim carries its epistemic label. Repo-derived numbers are reproducible from
`visualize_projection.py` at the current data pull (`_cc_decomp`,
`_cc_frontier_grade_algo`, `_cc_iso_compute`, `_cc_iso_compute_rate`,
`_cc_cn_crossing_sim`, `_pc_render_why`) and will drift as Epoch recomputes ECI.*
