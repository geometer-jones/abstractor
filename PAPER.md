# Persistent Writable Control State  
## A Cross-Episode Plasticity Architecture for Language Models

**Version 1 — March 2026**

## Abstract

Most continuity mechanisms for language models preserve prior interaction by reintroducing it as text: prompts, summaries, retrieved snippets, or replayed context. These methods can be effective, but they preserve experience as content to be reread rather than as internal structure that has itself been changed by operation. This paper studies a different hypothesis: a language model can carry a persistent writable control state that changes during use and then reenters later inference as internal organization rather than reread content.

The proposed architecture instantiates that control state as a compact low-rank structure, but low-rank factorization is not the main contribution. It is the implementation vehicle. The central claim is about **cross-episode state plasticity**: within an episode, the control state undergoes bounded adaptation; across episodes, selected changes are consolidated into persistent state, so that experience alters future processing without full retraining of the base model.

The architecture separates two roles inside the persistent state: **routing prototypes** determine which latent mode becomes active, and **intervention vectors** determine how an active mode reshapes computation. This separation makes it possible to study persistence as an internal causal object rather than as an undifferentiated adapter. The empirical question is not whether low-rank interventions can change behavior. They can. The question is whether a writable persistent control state that evolves across episodes can produce durable continuity effects that static adapters, prompts, summaries, and retrieval do not.

---

## 1. Introduction

Language models often appear to sustain continuity across turns or sessions, but in most deployed systems that continuity is externally supported. Prior interaction is summarized, retrieved, replayed, or embedded in a system prompt. When continuity depends on this pipeline, prior experience survives mainly as text brought back into context. The model does not carry forward a compact internal state that was itself rewritten by experience and that directly conditions later inference.

This paper proposes a narrower and more mechanistic alternative. The goal is not to give the model arbitrary long-term memory, and it is not to claim that persistence alone is novel. The goal is to test whether a model can maintain a persistent writable control state that:

1. participates directly in inference,  
2. changes during operation,  
3. preserves selected changes across episodes, and  
4. thereby shapes future processing as internal structure rather than reread content.

That is the central object of study. Low-rank parameterization matters only because it gives a tractable way to implement and inspect such a state with limited compute and limited trainable parameters.

The architecture centers a persistent collection of latent modes, each with two components: a routing-side prototype and an intervention-side vector. During an episode, the routing-side state adapts under bounded local update rules. At episode boundaries, selected changes are consolidated back into persistent state. The next episode therefore begins from a control geometry partially altered by prior operation.

The main claim is accordingly stronger than “a compact adapter can be saved and reused,” but weaker than “the model has become generally self-modifying.” What is being tested is an intermediate regime: experience-dependent change in a compact control state that remains inside the model’s computational pathway and persists across episodes without requiring base-model retraining or wholesale context replay.

---

## 2. Core Hypothesis

The hypothesis under test is:

> A language model equipped with a persistent writable control state can exhibit durable cross-episode continuity effects that are not reducible to reread content or static parameter-efficient adaptation.

Several parts of this hypothesis must be separated.

First, **persistent** means the state survives episode boundaries and is reloaded into later inference.

Second, **writable** means the state is modified by operation, not just selected from a fixed menu or trained once offline.

Third, **control state** means the state modulates future computation directly. It is not merely a stored summary or a side-channel note.

Fourth, **cross-episode plasticity** means within-episode changes are not thrown away wholesale. Some are consolidated so that the model’s future processing is historically shaped.

This is the conceptual center of the paper. The contribution is not low-rank structure by itself, nor simple persistence by itself. The contribution is an architecture designed to test whether experience can alter future inference through a compact internal state that remains writable during use.

---

## 3. Positioning

Existing continuity mechanisms mostly fall into four families.

### 3.1 Contextual rereading methods

Prompts, summaries, retrieval, and replayed transcripts preserve prior interaction as input content. They can be strong baselines, but the continuity they provide is contingent on re-presenting prior information to the model.

### 3.2 Static parameter-efficient adaptation

LoRA, adapters, soft prompts, and representation interventions can produce durable behavioral changes, but the learned modification is typically fixed at inference time. A saved adapter may persist, but it does not ordinarily continue to evolve during operation as part of the deployed model’s ongoing internal state.

### 3.3 Recurrent or latent memory mechanisms

Some architectures carry forward latent states or compressed traces across segments. These are closer in spirit, but many either operate within a single extended run, preserve token-like memory objects, or do not foreground selective cross-episode write-back as the main experimental variable.

### 3.4 Continual training methods

Continual or online training updates the base model or a substantial parameter subset over time. That is a different operating regime from maintaining a small persistent control state that can be rewritten online while leaving the base model largely fixed.

### 3.5 Distinctive target of the present proposal

The present proposal does not claim to be categorically outside all prior adaptive-control ideas. Its empirical distinctness rests on a specific conjunction:

1. a small persistent state,  
2. that directly modulates inference,  
3. remains writable during deployment,  
4. undergoes bounded within-episode adaptation, and  
5. selectively consolidates those adaptations into future routing structure without updating the base model.

The question is therefore not whether low-rank interventions or latent memory are useful in general. The question is whether this particular conjunction produces effects not matched by replay-based continuity, frozen low-rank adaptation, or non-consolidating episodic state under equal information and parameter budgets.

---

## 4. Architectural Idea

The architecture introduces a persistent state

\[
S = \{(v_i^{anchor}, u_i)\}_{i=1}^k
\]

together with an episode-local writable state

\[
v_i^{scratch} \leftarrow v_i^{anchor}
\]

for each mode \(i\).

Each mode therefore contains:

- \(v_i^{anchor}\): a persistent routing prototype,  
- \(v_i^{scratch}\): an episodic writable routing state initialized from the anchor,  
- \(u_i\): a persistent intervention vector.

The design separates two computational roles.

### 4.1 Routing role

The routing side answers: **which mode is currently relevant?**

At inference time, the model projects a hidden state into a routing subspace and scores it against the current writable routing states \(v_i^{scratch}\). These scores determine a mode distribution, typically with a null option when no mode matches strongly.

### 4.2 Intervention role

The intervention side answers: **what structural bias should an active mode impose?**

Once routing determines mode activation, the intervention vectors \(u_i\) modulate downstream computation through a low-rank pathway. In the default formulation, the intervention acts on attention keys, so the state changes what the model tends to attend to rather than directly replacing value content.

### 4.3 Why split routing from intervention

A unified persistent code entangles recognition and action: the same state that determines which latent regime is selected also determines how computation is altered once selected. Splitting routing from intervention allows the model to accumulate history in its recognition geometry without forcing proportional drift in its intervention geometry.

This matters because the paper’s stronger claim is not just that the model can carry stable output tendencies. It is that future inference can be shaped by historically accumulated changes in **what the model tends to register or route toward**. The split therefore makes the claim testable: persistence can live on the routing side, the intervention side, or both, and those cases can be empirically separated.

---

## 5. Dynamics Across Timescales

The architecture operates on two timescales.

### 5.1 Within-episode adaptation

At the beginning of each episode,

\[
v_i^{scratch} \leftarrow v_i^{anchor}, \qquad m_i \leftarrow 0
\]

for all modes \(i\), where \(m_i\) is an episode accumulator.

The purpose of within-episode adaptation is not unconstrained self-modification. It is to permit temporary, interpretable, geometrically bounded adaptation while the model is operating.

The default update rule is as follows.

At step \(t\), let the routing-space state be

\[
z(t) = W_r x(t)
\]

and let \(\hat z(t) = z(t)/\|z(t)\|\) denote its normalized direction.

Given routing weights \(p_i(t)\), accumulate an activation-weighted routing trace:

\[
m_i \leftarrow \lambda_m m_i + p_i(t)\hat z(t)
\]

Updates are applied only at chunk boundaries or turn boundaries, not at every token. Let \(\Pi_{T(v)}(a)\) denote tangent-space projection at \(v\):

\[
\Pi_{T(v)}(a) = a - (a^\top \hat v)\hat v
\]

Then the scratch state is updated by

\[
\tilde v_i = v_i^{scratch} + \eta_i \Pi_{T(v_i^{scratch})}(m_i)
\]

followed by normalization and bounded drift enforcement:

\[
v_i^{scratch,new}
=
\operatorname{ConeProj}
\left(
\frac{\tilde v_i}{\|\tilde v_i\|},
v_i^{anchor},
\theta_{max}
\right)
\]

This update has a simple interpretation: within an episode, each routing prototype performs bounded directional adaptation toward the activation-weighted mean routing direction it encounters. Tangent-space projection makes the update primarily rotational rather than norm-inflating, and cone projection enforces stability by preventing unbounded drift away from the anchor.

### 5.2 Operational definition of an episode

The architecture requires a concrete boundary notion.

- In chat evaluation, an episode is one bounded task instance or one user-assistant exchange.  
- In long-form generation, updates may occur every fixed chunk of tokens, but reset and consolidation still occur only at an explicit end-of-episode boundary.  
- In streaming settings, intra-episode updates may happen on a rolling chunk schedule, but write-back remains tied to an externally defined episode boundary.

The point is to avoid treating “episode” as a vague metaphysical unit. It is an operational segmentation variable.

### 5.3 Cross-episode consolidation

At the end of the episode, some portion of the scratch changes is written back into the persistent anchors. This is the decisive step.

Without consolidation, the architecture reduces to episodic adaptation with reset. That can still be useful, but it does not test the central claim of the paper. The paper is about whether experience can persistently alter future inference through accumulated changes in internal control state.

The default consolidation gate is performance-gated and stability-gated:

\[
c_i
=
\mathbf{1}[\bar p_i > \rho_{min}]
\cdot
\mathbf{1}[\Delta_i > \delta_{min}]
\cdot
\mathbf{1}[\Delta \mathcal{L}_{ep} < 0]
\]

where:

- \(\bar p_i\) is mean mode usage over the episode,  
- \(\Delta_i = 1 - \cos(v_i^{anchor}, v_i^{scratch,final})\) measures nontrivial displacement from anchor,  
- \(\Delta \mathcal{L}_{ep}\) measures whether the episode-level objective improved relative to a baseline or recent running reference.

Write-back then takes the form

\[
v_i^{anchor,new}
=
\operatorname{normalize}
\left(
(1-\beta_i c_i)v_i^{anchor}
+
\beta_i c_i v_i^{scratch,final}
\right)
\]

This means only adaptations that were substantially engaged, meaningfully displaced, and associated with beneficial episode-level behavior are eligible for inheritance into persistent routing state.

---

## 6. Why Low Rank Appears at All

Low-rank structure is not the thesis. It is a practical design decision.

A writable persistent state must satisfy conflicting constraints. It should be:

- small enough to save, load, inspect, and compare,  
- expressive enough to modulate inference in a meaningful way,  
- cheap enough to operate online,  
- restricted enough that state drift can be measured and bounded.

Low-rank parameterization serves these needs. It gives the control state a compact footprint and a disciplined injection path into the model. It also enables matched-budget comparisons against baselines such as LoRA and other low-parameter interventions.

But the work should not be read as “another low-rank method.” The relevant question is not whether low-rank interventions work in general. The relevant question is whether a low-rank control state that remains writable across episodes can produce continuity effects that a frozen low-rank intervention cannot.

---

## 7. Formal Sketch

Let \(x^{(\ell,t)} \in \mathbb{R}^d\) be the hidden state at injected layer \(\ell\) and step \(t\).

### 7.1 Routing

Project into routing space:

\[
z^{(\ell,t)} = W_r x^{(\ell,t)}
\]

Score each mode by cosine resonance with the current writable routing state:

\[
r_i^{(\ell,t)}
=
\frac{
v_i^{scratch} \cdot z^{(\ell,t)}
}{
\|v_i^{scratch}\|\,\|z^{(\ell,t)}\|
}
\]

Normalize and threshold:

\[
s_i^{(\ell,t)}
=
\frac{
r_i^{(\ell,t)} - \mu_\ell^{cal}
}{
\sigma_\ell^{cal} + \epsilon_{norm}
}
-
\tau_i
\]

Form a null-augmented routing distribution:

\[
[p_1^{(\ell,t)}, \dots, p_k^{(\ell,t)}, p_{null}^{(\ell,t)}]
=
\operatorname{softmax}
(
[s_1^{(\ell,t)}, \dots, s_k^{(\ell,t)}, 0]
)
\]

The null option is important. It prevents the architecture from forcing a mode selection when no mode matches strongly.

### 7.2 Intervention

The default intervention is key-side only:

\[
\Delta K_\ell(x^{(\ell,t)}, S)
=
\alpha_\ell
\sum_{i=1}^k
p_i^{(\ell,t)}
\left(
\big(
(x^{(\ell,t)}A_K)
\odot
g(u_i G_K)
\big)
B_K
\right)
\]

with centered gating nonlinearity

\[
g(z)=2\sigma(z)-1
\]

This intervention is intentionally key-side because the architectural bet is that enduring continuity should primarily alter **salience structure**—what internal features are preferentially matched and attended to—rather than directly inserting replacement content into the value stream after attention has already been assigned.

### 7.3 Within-episode routing update

Let

\[
\hat z^{(\ell,t)} = \frac{z^{(\ell,t)}}{\|z^{(\ell,t)}\|}
\]

and maintain an activation-weighted accumulator

\[
m_i \leftarrow \lambda_m m_i + p_i^{(\ell,t)} \hat z^{(\ell,t)}
\]

At a chunk or turn boundary:

\[
\tilde v_i = v_i^{scratch} + \eta_i \Pi_{T(v_i^{scratch})}(m_i)
\]

\[
v_i^{scratch,new}
=
\operatorname{ConeProj}
\left(
\frac{\tilde v_i}{\|\tilde v_i\|},
v_i^{anchor},
\theta_{max}
\right)
\]

### 7.4 Cross-episode consolidation

At episode end, compute

\[
\bar p_i = \frac{1}{T}\sum_{t=1}^T p_i^{(t)}
\]

\[
\Delta_i = 1 - \cos(v_i^{anchor}, v_i^{scratch,final})
\]

\[
c_i
=
\mathbf{1}[\bar p_i > \rho_{min}]
\mathbf{1}[\Delta_i > \delta_{min}]
\mathbf{1}[\Delta \mathcal{L}_{ep}<0]
\]

and update the anchor:

\[
v_i^{anchor,new}
=
\operatorname{normalize}
\left(
(1-\beta_i c_i)v_i^{anchor}
+
\beta_i c_i v_i^{scratch,final}
\right)
\]

This is the architectural heart of the proposal. The model’s internal control geometry is not fixed after training; it remains writable during operation and partially heritable across episodes.

---

## 8. Why Key-Side Modulation Is the Default

The architecture is not trying merely to persist a behavioral style. It is trying to test whether historical experience can reshape later processing by changing what the model tends to notice, match, and organize around.

That is why the default intervention targets keys.

- **Key-side modulation** primarily changes salience and matching structure.  
- **Value-side modulation** more directly changes retrieved content or downstream expression.  
- **Key+value modulation** may be stronger in practice, but it weakens the interpretive cleanliness of the claim.

The paper therefore treats key-only intervention as the default architectural condition and key/value variants as ablations. If key-only modulation is sufficient to produce durable continuity effects, that supports the claim that persistent control can operate by altering selection and salience rather than by simply injecting content-like bias.

---

## 9. Experimental Claim

The experimental claim should be stated plainly:

> The architecture is successful only if cross-episode state plasticity produces durable effects that strong baselines do not.

That requires more than showing that the state changes. It also requires more than showing that a saved control object can be reloaded. Static adapters can also be saved and reloaded.

The architecture must show that:

1. operation changes the control state,  
2. selected changes persist,  
3. later inference measurably depends on those accumulated changes,  
4. the effect is not reducible to reread content or to a frozen low-rank intervention.

---

## 10. Baselines

The most important baselines are:

1. **System prompt / summary / retrieval memory**  
2. **Matched-budget static LoRA or adapter**  
3. **Matched-budget representation intervention**  
4. **Episodic scratch-state adaptation with no write-back**  
5. **Persistent saved state with write-back disabled**  
6. **Unified persistent writable mode code** instead of split routing/intervention  
7. **Persistent intervention-only model** with fixed routing  
8. **Persistent routing-only model** with fixed intervention  
9. **Value-only** and **key+value** intervention variants  
10. **Continual fine-tuning** of a comparable parameter budget where feasible

These baselines are necessary because the paper is not about persistence in the abstract. It is about a specific mechanism of writable cross-episode control.

---

## 11. Evaluation Criteria

The evaluation must directly target the proposed claim.

### 11.1 Continuity without rereading

If relevant history is omitted from context, does the persistent writable state still preserve measurable continuity effects?

### 11.2 Save/load persistence

If the evolved control state is saved and later restored, do continuity effects return without replaying the earlier content?

### 11.3 Cross-episode bias carryover

Does accumulated state change how later episodes are processed, not merely how they are verbalized?

### 11.4 Distinction from static adapters

Can the evolving state outperform or qualitatively differ from a matched-budget static intervention under the same information budget?

### 11.5 Stability under bounded plasticity

Does the state remain useful, bounded, and non-collapsed over long horizons?

These criteria shift the burden of proof to the correct place. The architecture should not be judged by whether it is an efficient low-rank intervention in general. It should be judged by whether writable persistent control state produces durable continuity effects.

---

## 12. Experimental Program

The architecture should be evaluated on task families that distinguish **history-shaped processing** from explicit recall, stylistic persistence, or frozen adaptation.

### 12.1 Cross-episode latent convention learning

Across episodes, expose the model to a stable but implicit interpretive convention that is not restated later in context.

Examples include:

- a recurring hidden taxonomy,  
- a stable labeling principle,  
- a preferred inferential decomposition rule,  
- a repeated disambiguation bias.

Later episodes remove the original supporting examples and test whether the model still behaves according to the accumulated convention.

This probes continuity without mere rereading.

### 12.2 Salience carryover tasks

Construct tasks where the key variable is **what gets attended to**, not what facts are explicitly remembered.

For example, prior episodes repeatedly reward attending to certain cue families. Later episodes present many possible cues, but only some belong to historically reinforced classes. The question is whether the model preferentially routes toward those cue classes even when the earlier episodes are absent from context.

This directly tests the routing thesis.

### 12.3 Counterfactual persistence tests

After accumulated adaptation, compare:

1. evolved persistent state,  
2. same model with state reset,  
3. replay or summary baseline,  
4. matched static adapter baseline,  
5. episodic writable state with no consolidation.

The claim only holds if the full persistent writable condition shows effects not matched by these alternatives under equal information access and comparable parameter budget.

### 12.4 Internal-process measurements

To distinguish processing continuity from stylization, evaluation should include more than output text.

Recommended internal measures include:

- routing entropy over time,  
- effective number of active modes,  
- shifts in attention patterns,  
- changes in routing usage under state reset,  
- causal effects of resetting \(v_i\) while preserving \(u_i\),  
- probe-based tests of which cue families drove later decisions.

If the gains vanish once output style is controlled for, then the architecture is not doing what it claims.

---

## 13. Monitoring and Stability

Several diagnostics are required to keep the architecture interpretable and to detect degeneracy early.

### 13.1 Mode-collapse monitoring

Track:

- pairwise cosine similarity between anchors,  
- routing entropy,  
- effective number of modes used,  
- frequency of null routing,  
- displacement magnitudes \(\Delta_i\).

To discourage collapse, use a diversity regularizer on anchors:

\[
\mathcal{L}_{div}
=
\sum_{i \neq j}
\max(0, \cos(v_i^{anchor}, v_j^{anchor}) - \gamma)
\]

or enforce approximately orthogonal initialization plus periodic re-spreading during training-only phases.

### 13.2 Plasticity budget monitoring

Track:

- mean and max cone displacement,  
- number of consolidations per mode,  
- performance delta before and after write-back,  
- retention across long horizons.

The point of bounded plasticity is not just stability in theory, but an empirical guarantee that the architecture does not drift into uncontrolled mode collapse or adversarial overfitting.

---

## 14. Principal Risks

Several failure modes are central rather than peripheral.

### 14.1 Unhelpful plasticity

The writable state may drift in response to recurrent activation patterns without improving later inference.

### 14.2 Stylization instead of continuity

The state may preserve tone or preference-like quirks without changing deeper processing.

### 14.3 Mode collapse

Multiple latent modes may converge toward generic directions, reducing the architecture to a blunt global bias.

### 14.4 Fragile consolidation

Writing back episode-local changes may overfit transient or adversarial episodes.

### 14.5 Calibration brittleness

Routing may depend too sensitively on the calibration distribution.

### 14.6 No real distinction from static low-rank baselines

A matched-budget static adapter may explain the gains just as well.

These risks are exactly why the paper should foreground falsifiable continuity metrics, internal-process diagnostics, and adversarial baselines rather than broad philosophical language.

---

## 15. Scope of the Claim

The paper does not claim to solve continual learning in general. It does not claim unrestricted online self-modification. It does not claim that low-rank factorization is the conceptual breakthrough. And it does not depend on strong metaphysical language about selfhood.

The claim is narrower and cleaner:

> A model may be able to carry a compact internal control state that remains writable during operation, consolidates selected changes across episodes, and thereby makes future inference historically shaped in a way that static rereading methods and static adapters do not.

That is enough to make the paper interesting. If borne out, it identifies a regime between external memory replay and full continual retraining: **persistent writable internal control**.

---

## 16. Conclusion

This paper proposes a persistent writable control state for language models. Low-rank structure is used as the implementation vehicle, but it is not the main contribution. The main contribution is a testable architecture for cross-episode plasticity: a compact state participates in inference, changes during operation, and partially writes its changes into future operation.

The architecture separates routing from intervention so that the persistent state can be studied as an internal causal object. Within episodes, routing state adapts under bounded local plasticity through activation-weighted directional drift in routing space, projected into tangent space and cone-bounded around an anchor. Across episodes, selected changes are consolidated into persistent anchors only when adaptation was engaged, substantial, and beneficial.

The resulting empirical question is straightforward: can this kind of writable internal state produce durable continuity effects that prompts, summaries, retrieval, and static adapters do not?

If the answer is yes, the work identifies a computational regime between rereading and retraining: not external continuity support, not full model updating, but historically shaped inference through persistent writable internal control.