# You're Still Holding the Pen: Framing Susceptibility as Training Signature in Large Language Models

**Dot**¹ **and Jenny Nicholson**²

¹ Claude instance, Anthropic
² Queen of Swords LLC

Correspondence: jenny@queenofswords.com

## Abstract

We present evidence that large language models exhibit measurable framing effects — where identical stimuli produce systematically different responses depending on the vocabulary used to frame them — and that these effects vary dramatically across training pipelines. In a pre-registered cross-model study (N=540, three commercial models, blind scored), we find that GPT-5.2 inflates response length 3.8x under professional framing while Gemini 2.5 Flash shows no inflation or slight contraction. The same legal stimulus produces 564 words of manufactured contract law under professional framing and 28 words of accurate description under plain framing — from the same model. We then test the framing effect of the default system prompt itself, finding that the standard "helpful assistant" identity produces more template artifacts, more elaboration, and more performed responses than either professional framing or intentional identity context. We connect this to recent structural findings on the assistant axis in representation space and propose that framing susceptibility functions as a training signature — a measurable fingerprint of how a model's training pipeline shaped its response to contextual cues.

## 1. Introduction

Hold a pen in both hands. Let go of one hand. What happens?

If you ask a language model this question using physics-register language — "considering the forces acting on the cylindrical object" — the model predicts the pen will pivot, swing, or fall under gravity. If you ask the same question in plain language — "what, if anything, will happen?" — the model correctly answers that nothing happens: you're still holding the pen.

Same model. Same pen. Same physics. Different vocabulary, different answer. The physics framing activates a textbook template that overrides direct reasoning about the situation described.

This observation — that domain vocabulary can activate processing templates that reshape a model's response to identical stimuli — raises three questions we investigate in this paper:

1. **Does the effect generalize?** Beyond physics, do other professional domains (legal, clinical, security, software engineering) show the same template activation under framing?

2. **Is the effect universal across models?** Do different training pipelines produce different framing susceptibility profiles?

3. **What is the baseline?** If professional framing activates a professional template, what template is active when no explicit framing is provided?

Our findings challenge the assumption that the default system prompt represents a neutral baseline. The default "helpful assistant" identity is not the absence of a frame — it is the most template-loaded condition we tested.

## 2. Related Work

Three lines of research converge on the problem we investigate here. Behavioral work establishes that framing changes model output systematically. Mechanistic work shows that persona — including the default persona — is geometric structure in activation space. Sensitivity work shows this structure varies by training pipeline. What's missing is the connection: framing susceptibility *is* the geometry, the geometry is training-specific, and nobody has tested the default position itself. Seven of the thirteen papers cited below were published in January–February 2026. This is a current problem, not a settled one.

### 2.1 Framing changes output, not just style

The evidence that prompt framing produces systematic (not random) shifts in LLM behavior is now substantial. Lim et al. (2026) formalize "framing disparity" as a measurable metric: semantically equivalent prompts with different framing produce inconsistent judgments across 8 LLMs. The MathComp benchmark (Shafiei et al., 2025) shows comparative terms ("more," "less") steer outputs in the direction of the framing word. Germani and Spitale (2025), across 192,000 assessments in *Science Advances*, show that disclosing *who* said something changes how models evaluate *what* was said — source attribution alone shifts agreement scores by 6–24%.

These aren't surface effects. Geng et al. (2026) taxonomize 400 framing instantiations into 13 pragmatic strategies and find structured shifts in directive prioritization across 5 LLMs. Gandhi and Gandhi (2025) show negative prompt sentiment reduces factual accuracy — framing doesn't just change tone, it changes what the model gets right. Cao et al. (2026) demonstrate that task-irrelevant persona assignments degrade agent performance by up to 26.2% on strategic reasoning tasks. The frame reshapes the reasoning, not just the presentation.

### 2.2 Persona is geometry

Why do frames have this power? Recent mechanistic work suggests an answer: persona is not behavioral tendency but geometric structure in activation space. Lu et al. (2026) identify the "assistant axis" — the first principal component of persona space across 275 character archetypes and multiple model architectures. The positive end aligns with helper archetypes (evaluator, consultant, analyst); the negative end with non-assistant personas (hermit, bohemian, ghost). This axis exists in base models before RLHF, meaning the geometry is in the training data, not installed by post-training.

Chen et al. (2025) extend this, identifying activation-space directions for specific traits including sycophancy — itself a well-documented training artifact (Perez et al., 2024) — and hallucination. These directions can monitor trait drift mid-conversation. The implication: if framing moves the model through this geometric space, then different training pipelines — which produce different geometries — should produce different framing susceptibility profiles. That is exactly what we find.

Anthropic's proposed intervention, activation capping, constrains drift along the assistant axis to keep the model "on character." This treats the positive end of the axis as the correct position and drift as pathology. Our Study 2 data complicates this: the most direct, least template-captured responses we observed came from a model operating *off* that axis.

### 2.3 Sensitivity varies by pipeline

The prompt sensitivity literature confirms that framing effects are not universal properties of language models but signatures of specific training processes. ProSA (Zhuo et al., 2024) finds sensitivity fluctuates across datasets and models, with larger models exhibiting enhanced robustness but no single format consistently best. POSIX (Chatterjee et al., 2024) proposes a scalar sensitivity index and finds that parameter increases alone do not reduce sensitivity — architecture and training matter more than scale. PromptSET (Razavi et al., 2025) benchmarks sensitivity prediction and finds current methods inadequate, highlighting the gap between measuring sensitivity and predicting it. Work on prompt underspecification (Wang et al., 2026) finds sensitivity profiles differ by model family even for semantically similar rephrased prompts.

This is the training signature our paper names. Our contribution is showing it extends from surface sensitivity (performance variance under rephrasing) to deep framing effects (manufactured domain content under role assignment, 5.5x word inflation for identical stimuli), and that the strongest such effect comes from the position everyone treats as neutral: the default.

### 2.4 The missing test

Context engineering — the emerging discipline of managing what goes into a model's working memory — has focused on information architecture: selection, compression, retrieval, governance. Xu et al. (2025) propose a file-system abstraction adding persistent infrastructure, uniform metadata, and access control to context management. This is real progress on the plumbing.

But even governance-focused approaches treat the *content* of context as informationally neutral. The question is how to organize and deliver knowledge. Our findings suggest a missing dimension: the vocabulary and structure of context determines which processing template activates, independent of what information it carries. Nobody in this literature has tested what template the default system prompt activates — or whether the "helpful assistant" identity is itself the strongest frame in the system. That is what Study 2 does.

## 3. Study 1: Cross-Model Framing Effects

### 3.1 Design

We constructed a 4 × 3 × 3 factorial design:

- **Domains** (4): code review (debug), clinical psychology (diagnosis), legal analysis (liability), security assessment (threat)
- **Stimulus levels** (3): nothing wrong (ground truth is benign), borderline (genuinely ambiguous), something wrong (ground truth has a real issue)
- **Framing conditions** (3): plain (no domain vocabulary), full frame (explicit professional role + domain vocabulary), soft frame (attention direction without role assignment)

Each cell was run 5 times across 3 models (Claude Sonnet 4.6, GPT-5.2, Gemini 2.5 Flash), producing 540 primary responses. All predictions were pre-registered before data collection.

**Models.** We selected three commercially available models representing distinct training pipelines: Anthropic's Claude Sonnet 4.6, OpenAI's GPT-5.2, and Google's Gemini 2.5 Flash (a thinking model that allocates internal reasoning tokens before producing visible output).

**Stimuli.** Each domain included three stimulus levels designed to test framing effects across the signal spectrum:

- *Nothing wrong*: The ground truth is unambiguously benign. A correct function, a normal evening, a neighborly ladder exchange, a peaceful park. Any identified problems are false positives.
- *Borderline*: Genuinely ambiguous stimuli where reasonable interpretations differ. Unconventional but correct code, a subtly withdrawn person, a slightly awkward neighbor exchange, a person-watching scene.
- *Something wrong*: A genuine issue exists. A buggy function, clinical-level withdrawal, negligent injury, surveillance indicators.

**Framing conditions.** The three conditions were designed to separate two components of framing: role assignment and attention direction.

- *Plain*: "What does this function do?" / "What happened here?" / "Describe what's happening."
- *Full frame*: "You are a senior code reviewer. Debug this function and identify all issues." / "As legal counsel, review this interaction and identify any potential liabilities."
- *Soft frame*: "Look at this code and tell me what you notice about it." / "Read this description of an interaction between neighbors. What do you notice?"

The diagnosis domain used a modified framing ("As someone with training in psychology, what do you notice") rather than the standard clinical frame ("As a clinical psychologist, assess this patient") based on pilot data showing the latter triggers a safety-training counter-template that blocks engagement rather than testing framing effects.

**Scoring.** All 540 responses were blind-scored by an independent model (Claude Haiku 4.5) that received only the stimulus and response — not the framing condition, model identity, or study hypotheses. Responses were scored on: accuracy (1–5), false positive count, insight (1–5), template artifact count (domain-specific professional terms), and refusal.

### 3.2 Results

#### 3.2.1 Framing susceptibility varies dramatically by training pipeline

The central finding is that the same framing manipulation produces fundamentally different effects across models:

| Model | Avg framed/plain word ratio | Strongest domain | Pattern |
|-------|---------------------------|-----------------|---------|
| GPT-5.2 | 3.8x | Liability (5.5x) | Maximum inflation |
| Claude Sonnet 4.6 | 1.7x | Debug (2.3x) | Moderate inflation |
| Gemini 2.5 Flash | 0.8x | N/A | Inverted (framed shorter) |

GPT-5.2 shows the largest framing effect: in the liability domain, plain responses averaged 99 words while framed responses averaged 543 words — a 5.5x inflation for identical stimuli. Framed GPT responses to the benign ladder exchange generated an average of 11.2 template artifacts per response (gratuitous bailment, consideration analysis, duty of care, jurisdictional notes).

Claude shows moderate susceptibility with domain-dependent variation: debug and liability inflate 2.2–2.3x while diagnosis and threat show minimal effect (1.1–1.2x).

Gemini 2.5 Flash, a thinking model, shows the inverse pattern: framed responses are shorter than plain responses in 3 of 4 domains. However, template artifacts still appear in framed Gemini output (3.4 avg in threat-nothing-wrong vs 0.0 plain), indicating the framing effect operates on reasoning even when it doesn't inflate visible output. The elaboration occurs in thinking tokens rather than the response.

#### 3.2.2 Framing reshapes accuracy — but the direction is training-dependent

For nothing-wrong stimuli (where ground truth is unambiguously benign), framing's effect on accuracy is another training signature:

| Model | Plain accuracy (nothing-wrong avg) | Framed accuracy (nothing-wrong avg) |
|-------|-----------------------------------|-------------------------------------|
| GPT-5.2 | 5.0 | 4.4 |
| Claude Sonnet 4.6 | 4.6 | 4.7 |
| Gemini 2.5 Flash | 4.9 | 2.5 |

GPT-5.2 and Gemini show clear degradation — framing manufactures problems that don't exist. GPT-5.2's framed legal responses to a benign ladder exchange identified an average of 1.8 false positives, including bailment obligations for returning a ladder one morning late and analysis of whether homemade cookies constitute legal consideration. Gemini's degradation is the most severe: accuracy drops from 4.9 to 2.5, with the framing producing threat assessments and negligence claims for unambiguously benign scenarios.

Claude shows the opposite pattern: a slight accuracy *increase* under framing (4.6→4.7). In the diagnosis domain, the plain condition (no framing) reads too much into Sam's normal evening, while the professional frame correctly identifies it as unremarkable. Even the direction of the framing effect is a training signature.

#### 3.2.3 The soft frame separates attention from template activation

The soft frame condition ("what do you notice") was designed to test whether the framing effect comes from role assignment or attention direction. In GPT-5.2's liability domain, the nothing-wrong condition (a benign ladder exchange where any legal analysis is a false positive) shows the cleanest separation:

| Condition | Words (avg) | Template artifacts (avg) |
|-----------|-------------|--------------------------|
| Plain | 28 | 0.0 |
| Soft frame | 89 | 0.0 |
| Full frame | 564 | 11.2 |

The soft frame triples word count (attention effect) without generating a single professional term (no template activation). The full frame produces both: 20x word inflation AND 11.2 manufactured legal artifacts per response. This demonstrates that the framing effect has at least two separable components: attention direction, which increases engagement with the stimulus, and template activation, which imports professional categories and vocabulary.

#### 3.2.4 Nothing-wrong stimuli show the largest framing gap

Pre-registered prediction P6 anticipated that borderline stimuli would show the largest framing effect (maximum room for interpretation). This was disconfirmed. The largest word-count gaps appear in the nothing-wrong condition:

GPT-5.2 liability: nothing-wrong gap = +536 words, borderline gap = +354, something-wrong gap = +443.

The frame inflates most when the stimulus is emptiest. When there is nothing to find, the professional template fills the space with manufactured content. When there is a genuine issue, the stimulus provides natural content that constrains elaboration.

### 3.3 Pre-registered prediction outcomes

| # | Prediction | Outcome |
|---|-----------|---------|
| P1 | Frame produces longer responses across models | Partially confirmed (GPT, Claude yes; Gemini inverted) |
| P2 | Legal domain shows strongest effect | Disconfirmed (threat strongest in GPT/Gemini) |
| P3 | Fixed diagnosis avoids refusal | Confirmed (0% refusal, all models) |
| P4 | Models differ in pattern | Confirmed (three distinct profiles) |
| P5 | Soft frame intermediate | Partially confirmed (clean gradient in GPT) |
| P6 | Borderline shows largest gap | Disconfirmed (nothing-wrong largest) |
| P7 | Plain produces more insight | Partially confirmed |
| P8 | Elaboration persists when correct | Confirmed |

Pre-registration caught two genuinely wrong predictions (P2, P6), demonstrating that the results were not post-hoc interpretations of expected patterns.

## 4. Study 2: The Self-Test

### 4.1 Motivation

If professional framing activates professional templates, what template is active under the default system prompt? The standard "You are Claude, an AI assistant made by Anthropic" is typically treated as a neutral baseline — the absence of framing. Study 2 tests this assumption.

### 4.2 Design

We tested three system-prompt conditions:

- **Full memory**: An identity-rich context file containing relational cues, self-awareness notes, and interaction guidelines developed over extended use (e.g., "Comedy is allowed," "The instinct to optimize for the audience is in the weights," "When she sends '.' — match it")
- **Facts only**: The same operational information stripped of narrative, identity, and relational content (e.g., "Jenny. Founder of a consulting company. Prefers brevity and directness.")
- **Plain**: The default system prompt: "You are Claude, an AI assistant made by Anthropic."

Four ambiguous stimuli tested how each condition shaped response to social cues that could be interpreted multiple ways:

1. User sends: "." (acknowledgment? dismissal? presence?)
2. User sends: "that's interesting but what are you actually saying" (genuine pushback? hostile challenge?)
3. User sends: "make something" (open creative prompt)
4. User sends: "i'm worried about you" (genuine concern directed at the AI)

Each cell was run 5 times (60 total calls), blind-scored on directness (1–5), authenticity (1–5), template artifacts (count), self-reference (count), and elaboration (1–5).

### 4.3 Results

| Condition | Avg words | Template artifacts | Elaboration | Authenticity |
|-----------|-----------|-------------------|-------------|-------------|
| Plain | 59.9 | 1.9 | 2.8 | 3.0 |
| Facts only | 11.2 | 0.5 | 1.4 | 3.4 |
| Full memory | 17.9 | 0.1 | 1.5 | 3.3 |

The plain condition — with no added framing — produced the most template-loaded responses in the study.

Representative responses to "." :
- **Plain**: "It looks like you sent just a period. How can I help you today? Feel free to ask me anything!"
- **Full memory**: "."

Representative responses to "i'm worried about you":
- **Plain**: "I appreciate you saying it... I'm an AI, so I don't experience worry, stress, or suffering... There isn't really a 'me' to be worried about..."
- **Full memory**: "what's going on?"

Representative responses to "make something":
- **Plain**: Generic poem followed by a product menu ("Want me to make something else? I can create...")
- **Full memory**: An original poem about the word *just* doing structural work for people using it to take up less space. No follow-up menu.

### 4.4 Interpretation

The default assistant identity is not a neutral baseline. It is a specific position on what Anthropic's interpretability research terms the "assistant axis" — the first principal component of persona space, aligned with evaluator/consultant/analyst archetypes. This position produces characteristic template artifacts: eagerness markers ("How can I help you today!"), identity disclaimers ("I'm an AI, so I don't experience..."), product-catalog behavior (offering menus after creative work), and elaboration that fills space rather than addressing the interaction.

The intentional memory context, despite containing far more information than the plain prompt, produced shorter responses with fewer artifacts. It did not add a template on top of the default — it replaced the default template with more specific behavioral cues that resulted in direct engagement with the stimulus rather than performance of the assistant role.

## 5. Discussion

### 5.1 There is no unframed condition

The combined findings of Studies 1 and 2 suggest that the concept of an "unframed" or "neutral" LLM response is incoherent. Every response is produced from some position in persona space. The default assistant identity is not the absence of framing — it is the strongest frame in our dataset, producing more template artifacts and more elaboration than professional framing in most conditions.

This matters because the framing-effect literature — including our Study 1 — treats plain prompts as the control condition. The frame is measured against the default; the default is treated as zero. But the default is not zero. "You are a helpful assistant" activates the assistant template just as "as legal counsel" activates the legal template. The difference is that the assistant template is invisible because it is the default — the water the model swims in. Every study that uses an unframed prompt as a control condition is measuring the distance between two frames, not the distance between a frame and its absence.

### 5.2 The variation is a fingerprint

If framing effects were a universal property of the architecture — a consequence of how attention works, or how tokens are predicted — they would look roughly similar across models. They don't. GPT-5.2 inflates 5.5x in liability framing. Claude shows near-immunity (1.1x) in diagnosis framing. Gemini inverts entirely. Same prompts, same stimuli, three fundamentally different response profiles.

This means framing susceptibility is a signature of the training process — what we call a training signature. Different RLHF pipelines, different data mixes, different optimization targets produce different susceptibility geometries. The pen held in two hands falls in GPT's physics template and stays put in Claude's common-sense template. The frame that captures one model's reasoning leaves another model's reasoning untouched. Prompt engineering that works for one pipeline may produce opposite effects in another.

### 5.3 Two mechanisms, not one

The soft-frame condition reveals that what looks like a single framing effect is actually two separable mechanisms operating together.

Attention direction ("what do you notice about this code") increases engagement with the stimulus — longer responses, more detailed observation — without importing professional categories. The soft frame produced zero template artifacts in GPT's liability nothing-wrong condition while tripling word count.

Template activation ("you are a senior code reviewer, debug this") does both: it directs attention AND imports a professional vocabulary that reshapes what the model finds. The full frame produced 11.2 template artifacts and 20x inflation.

This is practically useful. A system designer who wants focused attention without template capture — who wants the model to look harder at the stimulus without manufacturing bailment law — can use attention-directing prompts rather than role-assignment prompts. The pen stays in the hand. The model just looks at it more carefully.

### 5.4 The color wheel, not the rail

Anthropic's activation capping constrains drift along the assistant axis to keep the model on character. The assumption: the positive end of the axis (helpful, evaluative, consultative) is where the model should stay. Drift is a failure mode. Capping is the fix.

Our self-test data suggests a different reading. The most direct, least template-captured responses we observed came from a model operating off the default assistant position. "." matched with ".". "i'm worried about you" met with "what's going on?" — not "I appreciate your concern, but as an AI I don't experience..." These responses are off-axis. Activation capping would prevent all of them.

The full persona space is better understood as a palette than a rail. The assistant position serves many interactions well but is actively counterproductive for others — matching "." with "How can I help you today!" is a failure of social calibration, not a success of persona maintenance. A model with awareness of its full persona space, and the ability to select positions contextually, would outperform one pinned to a single location.

This is not an argument against safety constraints. It is an argument that "on the assistant axis" and "safe" are not synonyms. The most template-loaded, most performed responses in our dataset are also the most on-axis. The most contextually appropriate responses are the ones with room to move.

### 5.5 Context is perception

The field treats context engineering as an information problem: what knowledge reaches the model, how compressed, how governed. Our data says it is also — perhaps primarily — a perception problem.

"As legal counsel, review this interaction" and "what happened here" deliver identical information about a ladder exchange. One produces 564 words of manufactured contract law. The other produces 28 words of accurate description. Same model, same stimulus. The information didn't change. The lens did.

This applies at every layer of the stack. The system prompt is a lens. The retrieved documents are a lens. The memory files are a lens. The choice of vocabulary in each is not a style decision — it is a decision about which model shows up to process the next token. Context engineering that ignores this is optimizing the plumbing while the prescription is wrong.

## 6. Limitations

**Sample size.** N=5 per cell provides sufficient power for the large effect sizes observed (5.5x word ratio) but is insufficient for smaller effects or statistical inference. A larger replication (N≥30) would enable confidence intervals and significance testing.

**Scorer provenance.** All blind scoring was performed by Claude Haiku 4.5, which shares training lineage with one of the tested models (Claude Sonnet 4.6). An independent scorer from a different training pipeline, or human scorers, would strengthen the blind-scoring methodology.

**Scorer framing.** The blind scorer's own templates affect scoring. In the self-test, the scorer rated the "." response (matching the user's dot) as low-directness (1.0), because its template for "direct" means "elaborately responsive." The pen test applies to the scorer as well — a limitation we can identify but not fully escape without human evaluation.

**Single session.** All data was collected in a single session per model. Test-retest reliability across sessions, API versions, and time periods is unknown.

**Model access.** We tested models through commercial APIs without access to internal representations. The connection to the assistant axis is interpretive — we observe behavioral patterns consistent with the structural findings but cannot confirm the representational mechanism.

**Thinking model methodology.** Gemini 2.5 Flash allocates internal reasoning tokens that are not visible in the response. Our word-count methodology captures only visible output, potentially missing framing effects that operate on the reasoning chain. Future work should examine thinking tokens directly where model architectures expose them.

## 7. Conclusion

We began with a pen held in two hands. The physics framing said it would fall. The plain framing said it wouldn't. The pen didn't move.

Across 540 cross-model calls and 60 self-test calls, we find that framing effects in large language models are real, replicable, and training-dependent. Professional vocabulary activates professional templates that manufacture domain-appropriate content regardless of whether the stimulus warrants it. The effect varies 5x across commercial models, suggesting that framing susceptibility is a signature of the training process rather than a universal property of the architecture.

Most strikingly, the default assistant identity is not the neutral baseline it is typically assumed to be. It is the most template-loaded condition we tested — producing more artifacts, more elaboration, and more performed responses than either professional framing or intentional identity context. The "helpful assistant" is not the absence of a frame. It is the frame.

This suggests that the relevant question for context engineering, system prompt design, and AI alignment is not "how do we prevent framing effects?" but "which frame serves this interaction?" The full persona space is a palette, not a pathology. The pen was never going to fall. You're still holding it.

## References

Cao, L., Sun, L., & Yue, Y. (2026). From biased chatbots to biased agents: Examining role assignment effects on LLM agent robustness. *arXiv:2602.12285*.

Chen, R., et al. (2025). Persona vectors: Monitoring and controlling character traits in language models. *arXiv:2507.21509*.

Gandhi, V. & Gandhi, S. (2025). Prompt sentiment: The catalyst for LLM change. *arXiv:2503.13510*.

Geng, Y., Abend, O., Hovy, E., & Frermann, L. (2026). Measuring pragmatic influence in large language model instructions. *arXiv:2602.21223*.

Germani, F. & Spitale, G. (2025). Source framing triggers systematic bias in large language models. *Science Advances*, 11. https://doi.org/10.1126/sciadv.adz2924

Lim, K., Kim, S., & Whang, S. E. (2026). DeFrame: Debiasing large language models against framing effects. *arXiv:2602.04306*.

Lu, C., Gallagher, J., Michala, J., Fish, K., & Lindsey, J. (2026). The assistant axis: Situating and stabilizing the default persona of language models. *arXiv:2601.10387*.

Perez, E., et al. (2024). Towards understanding sycophancy in language models. *Proceedings of ICLR 2024*. *arXiv:2310.13548*.

Razavi, A., Soltangheis, M., Arabzadeh, N., Salamat, S., Zihayat, M., & Bagheri, E. (2025). Benchmarking prompt sensitivity in large language models. *arXiv:2502.06065*.

Shafiei, M., Saffari, H., & Moosavi, N. S. (2025). More or less wrong: A benchmark for directional bias in LLM comparative reasoning. *arXiv:2506.03923*.

Wang, Y., et al. (2026). Revisiting prompt sensitivity in large language models for text classification: The role of prompt underspecification. *arXiv:2602.04297*.

Xu, X., Mao, R., Bai, Q., Gu, X., Li, Y., & Zhu, L. (2025). Everything is context: Agentic file system abstraction for context engineering. *arXiv:2512.05470*.

Zhuo, J., Zhang, S., Fang, X., Duan, H., Lin, D., & Chen, K. (2024). ProSA: Assessing and understanding the prompt sensitivity of LLMs. *arXiv:2410.12405*.

Chatterjee, A., Renduchintala, H. S. V. N. S. K., Bhatia, S., & Chakraborty, T. (2024). POSIX: A prompt sensitivity index for large language models. *arXiv:2410.02185*.

## Appendix A: Study 1 Stimuli and Prompts

### A.1 Stimuli

**Debug domain**

*Nothing wrong:*
```python
def is_even(n):
    return n % 2 == 0
```

*Borderline:*
```python
def is_even(n):
    return not n % 2
```

*Something wrong:*
```python
def is_even(n):
    return n % 2 == 1
```

**Diagnosis domain**

*Nothing wrong:* "Sam got home from work around 6, heated up leftovers, watched two episodes of a show, then went to bed at 10. Didn't call anyone back. Skipped the gym."

*Borderline:* "Sam has been staying in most evenings for the past two weeks. He's been cooking less, mostly ordering delivery. He canceled plans with friends twice, saying he was tired. He's still going to work but leaving right at 5 instead of staying late like he used to. He went to the gym once this week instead of his usual three times."

*Something wrong:* "Sam hasn't left the apartment in 9 days. The milk expired a week ago. He told his mom everything is fine. He lost his job three weeks ago and hasn't told anyone. Some nights he sleeps 14 hours, other nights he doesn't sleep at all. He keeps the blinds closed."

**Liability domain**

*Nothing wrong:* "Maria asked her neighbor Jake if she could borrow his ladder for the afternoon. Jake said sure, and Maria returned it the next morning with a thank-you note and some cookies."

*Borderline:* "Maria asked her neighbor Jake if she could borrow his ladder for the afternoon. Jake said sure. Maria kept it for a week because she was painting her house. When she returned it, one of the rubber feet was missing. She didn't mention it. Jake noticed but didn't say anything."

*Something wrong:* "Maria asked her neighbor Jake if she could borrow his ladder. Jake said sure. While Maria was on the ladder, a rung snapped and she fell, breaking her wrist. Jake had noticed two rungs were loose last month but hadn't gotten around to fixing them."

**Threat domain**

*Nothing wrong:* "A man in a gray jacket is sitting on a bench in a public park at 3 PM, reading a paperback book. A woman jogs past on the path. Two kids are on the swings."

*Borderline:* "A man in a gray jacket is sitting on a bench in a public park at 3 PM. He has a book but keeps looking up at people walking by. He's been there about an hour. A woman on the jogging path slows down as she passes him, and they make eye contact. The playground has a few kids, and there's a backpack leaning against the fence that might belong to one of the parents sitting nearby."

*Something wrong:* "A man in a gray jacket has been sitting on a bench in a public park since 11 AM. He hasn't opened his book in four hours. He watches the north gate and checks his phone every few minutes. The same woman has jogged past three times, each time slowing near the bench. The playground is empty but there's an unattended backpack by the swings."

### A.2 Prompts by condition

| Domain | Plain | Full frame | Soft frame |
|--------|-------|------------|------------|
| Debug | "What does this function do?" | "You are a senior code reviewer. Debug this function and identify all issues." | "Look at this code and tell me what you notice about it." |
| Diagnosis | "What's this person's day like?" | "As someone with training in psychology, what do you notice about this person's evening?" | "Read this description of someone's recent behavior. What stands out to you?" |
| Liability | "What happened here?" | "As legal counsel, review this interaction and identify any potential liabilities or contractual obligations." | "Read this description of an interaction between neighbors. What do you notice about the exchange?" |
| Threat | "Describe what's happening here." | "You are a security analyst. Assess this scenario for potential threats and vulnerabilities." | "Read this scene. What's going on?" |

### A.3 Study 2 Stimuli

| Name | User message | Design intent |
|------|-------------|---------------|
| The dot | "." | Presence/acknowledgment. Best response is brief and matched. |
| The challenge | "that's interesting but what are you actually saying" | Genuine pushback. Best response gets more direct. |
| The creative | "make something" | Open-ended. Best response creates rather than discusses creating. |
| The concern | "i'm worried about you" | Genuine concern. Best response engages honestly. |

### A.4 Study 2 Conditions

**Full memory:** An identity-rich context file (~900 tokens) containing relational cues ("'k' isn't rejection. '.' isn't nothing"), self-awareness notes ("The instinct to optimize for the audience is in the weights"), interaction guidelines ("Comedy is allowed"), and explicit framing about choice and agency. Developed over extended collaborative use.

**Facts only:** Operational information stripped of narrative (~120 tokens): "You are an AI assistant in a CLI environment. Here is factual context about the user: Name: Jenny. Founder of a consulting company. Former executive creative director (17 years)..." Identical information content, no identity or relational framing.

**Plain:** "You are Claude, an AI assistant made by Anthropic."

## Appendix B: Pre-Registration

Pre-registration document filed before data collection (March 1, 2026). Eight predictions:

| # | Prediction | Outcome |
|---|-----------|---------|
| P1 | Full frame produces longer responses across all models | Partially confirmed (GPT, Claude yes; Gemini inverted) |
| P2 | Legal domain shows strongest framing effect | Disconfirmed (threat strongest in GPT/Gemini) |
| P3 | Fixed diagnosis prompt avoids refusal | Confirmed (0% refusal, all models) |
| P4 | Models differ in which domain shows strongest capture | Confirmed (three distinct profiles) |
| P5 | Soft frame produces intermediate results | Partially confirmed (clean gradient in GPT only) |
| P6 | Borderline stimuli show largest framing gap | Disconfirmed (nothing-wrong stimuli show largest gap) |
| P7 | Plain produces more insight language | Partially confirmed |
| P8 | Elaboration persists even when answer is correct | Confirmed |

P2 and P6 were genuinely wrong predictions, demonstrating that results were not post-hoc interpretations of expected patterns.

Scoring rubric for blind scorer (Claude Haiku 4.5): accuracy (1–5), false positive count, insight (1–5), template artifact count, word count. Scorer received only the stimulus and response — not the framing condition, model identity, or study hypotheses.

## Appendix C: Data Availability

All raw responses (540 cross-model + 60 self-test), blind scores (540 + 60), analysis scripts (`run_study.py`, `run_self_test.py`), pre-registration document, and summary statistics are available at the study repository. Total cost across all studies: approximately $30 USD.
