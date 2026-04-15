
# Lunisolar Common-Mode Forcing of QBO, ENSO, and Chandler Wobble: A Synthesis of Reported Results

## Executive summary

This white paper documents a completed line of analysis asserting that the quasi-biennial oscillation (QBO), El Niño-Southern Oscillation (ENSO), and Chandler wobble are all expressible as responses to long-period lunisolar forcing generated through annual aliasing of shorter tidal cycles, especially the draconic cycle and its harmonics.[^60][^63][^64] In this framework, the QBO and Chandler wobble serve as spectrally precise “lemma” cases, while ENSO and mean sea-level (MSL) records provide higher-dimensional confirmation that the same forcing family operates across oceanic and rotational observables.[^59][^60][^64]

The principal claim is not merely that lunar signals exist in geophysics, but that the detailed period structure, cross-validated phase fits, and multivariate coherence archived in the Mathematical Geoenergy program and GeoEnergyMath/GitHub analyses together support a common-mode forcing interpretation that has been under-recognized in mainstream climate and geophysical modeling.[^59][^60][^64] The novel physical synthesis advanced here is that draconic-forced ocean and sea-level variability can amplify equatorial Kelvin-wave activity, thereby reinforcing the atmospheric wave field that governs QBO momentum deposition and phase reversals.[^36][^41][^43]

The emphasis of this report is on what has already been derived, archived, and cross-validated in public sources, not on speculative future modeling.[^60][^64][^86] The open publication of formulas, figures, repositories, and review exchanges is itself central to the argument, because it allows other analysts to verify, refine, or reject the interpretation on a fixed evidentiary record rather than on shifting standards tied to any single general circulation model (GCM).[^5][^6][^73]

## Background and scope

The short ESD Ideas paper submitted to *Earth System Dynamics Discussions* stated the central thesis succinctly: three cyclic geophysical behaviors with no settled mechanistic consensus — ENSO, QBO, and Chandler wobble — may share a common long-period tidal forcing architecture tied to lunar nodal and annual interactions.[^60] That paper was intentionally brief, but it clearly laid out three linked propositions: ENSO as a forced oceanic sloshing response, QBO as a gravitationally triggered atmospheric reversal process, and Chandler wobble as a forced torque response of the rotating Earth.[^60]

The same paper also argued that the lack of consensus in each of the three fields made a common parsimonious forcing hypothesis scientifically worth considering.[^60] For the QBO specifically, it noted that equatorial zonal winds reverse on average every 14 months and suggested that aliasing of lunar nodal timing against the annual cycle supplies a natural route to the observed ~28 month full oscillation period.[^60]

The reviewer exchange attached to the ESDD discussion sharpened that argument in two important ways.[^62] First, it made explicit that the period-doubling from the Chandler wobble scale (~433 days) to the QBO scale (~2 × 433 days) follows from aliasing of the draconic nodal timing against the annual cycle rather than from ad hoc tuning.[^62] Second, it clarified that the ENSO spectrum can be populated by multiple lunar tidal components, including fortnightly and monthly terms whose close annual aliases and nonlinear combinations generate the broad range of interannual and multidecadal variability seen in ocean records.[^62]

This white paper takes that compact published statement and its interactive discussion as a starting point, then consolidates the more detailed evidence available in the Mathematical Geoenergy text, GeoEnergyMath posts, and open repositories.[^59][^60][^64]

## Core result

The central result documented across the archived analyses is that long-period geophysical variability can be modeled as deterministic or quasi-deterministic responses to alias products of well-defined lunisolar cycles, with annual modulation serving as the conversion mechanism from short astronomical periods to observed interannual or multiyear geophysical periods.[^60][^63][^64] The importance of this claim lies in the fact that the relevant frequencies are not chosen arbitrarily; they arise from fixed astronomical periods and their alias relationships.[^60][^63]

Within this framework, the QBO and Chandler wobble are especially important because their dominant periods are relatively sharp and therefore resistant to casual overfitting.[^59][^64] If the same forcing logic reproduces both a ~433 day rotational oscillation and a ~28 month equatorial stratospheric oscillation, then the common-mode argument is strengthened before the more complex ENSO case is even considered.[^59][^60][^63]

ENSO and MSL then provide a broader test of whether the same forcing family has predictive value in coupled ocean-atmosphere dynamics.[^60][^66][^86] The archived GEM-LTE results, related cross-validation posts, and supporting writeups are therefore not side examples; they are the main demonstration that the common-mode structure scales beyond narrow spectral matches into multi-index climate behavior.[^66][^83][^86]


##  Latent-Phase Equivalence of LTE and Classical β-Plane Dynamics  
### A Sequential Derivation and Cross-Validation

This section presents two *independent* derivations of the same latent dynamical structure:

1. A derivation from the **Local Tangent Equations (LTE)** in a rotating frame.
2. A derivation from the **classical β-plane shallow-water system**.

Both routes converge to the same reduced-order form:



$$
\Phi(t) = \sum_i k_i \sin(\omega_i t), \qquad 
\zeta(t) = \sin\Phi(t),
$$



where  
- Φ(t) is a **latent phase coordinate**,  
- ζ(t) is a **nonlinear observable**,  
- ω<sub>i</sub> are physical forcing frequencies (lunisolar, annual, etc.),  
- <i>k</i><sub>i</sub> are coupling coefficients determined by the reduction.

This equivalence is not assumed — it is *derived twice* from independent starting points.

SVG figures are referenced inline; ASCII fallback diagrams appear in the Appendix.

---

## 1. LTE-Based Derivation  
### 1.1 Starting point: rotating-frame LTE

We begin with the LTE formulation for a rotating fluid parcel on a sphere:



$$
\frac{d\mathbf{u}}{dt} + 2\boldsymbol{\Omega}\times\mathbf{u}
= -\nabla\Phi + \mathbf{F}(t),
$$



where \(\mathbf{F}(t)\) contains periodic astronomical forcing.

### 1.2 Coriolis cancellation along the equator

At the equator, the meridional component of the Coriolis term vanishes:



$$
f = 2\Omega\sin\varphi = 0 \quad \text{at } \varphi=0.
$$



This yields a scalar equation in longitude \(\lambda\) and time \(t\):



$$
\frac{\partial u}{\partial t} = -\frac{\partial \Phi}{\partial \lambda} + F(t).
$$



### 1.3 Reduction to a scalar PDE

Assuming separability in \(\lambda\):



$$
u(\lambda,t) = X(\lambda) T(t),
$$



we obtain a temporal equation of the form:



$$
\frac{dT}{dt} + \alpha T = F(t).
$$



### 1.4 Sturm–Liouville reduction

Periodic forcing \(F(t)\) admits a Fourier expansion:



$$
F(t) = \sum_i k_i \sin(\omega_i t).
$$



Solving the linear ODE yields:



$$
T(t) = \sum_i k_i \sin(\omega_i t + \phi_i).
$$



### 1.5 Latent phase coordinate

Define the **latent phase**:



$$
\Phi(t) = \sum_i k_i \sin(\omega_i t).
$$



### 1.6 Observable

A nonlinear observable (e.g., displacement, height anomaly, angular momentum) is:



$$
\zeta(t) = \sin\Phi(t).
$$



### 1.7 Figure: LTE operator reduction

![LTE operator reduction](/assets/media/lte_operator_reduction.svg)

---

## 2. Classical β-Plane Derivation  
### 2.1 Shallow-water β-plane equations

The linearized shallow-water system:



$$
\begin{aligned}
u_t - \beta y v &= -g\eta_x, \\
v_t + \beta y u &= -g\eta_y, \\
\eta_t + H(u_x + v_y) &= 0.
\end{aligned}
$$



### 2.2 Meridional Sturm–Liouville eigenproblem

Assume modal structure:



$$
(u,v,\eta)(x,y,t) = \Re\{A(t)\phi(y)e^{ikx}\}.
$$



This yields the meridional eigenproblem:



$$
\mathcal{L}\phi = \lambda \phi,
$$



with Hermite-like solutions for equatorial modes.

### 2.3 Single-mode reduction

Projecting onto a single equatorial mode gives:



$$
\frac{dA}{dt} + \gamma A = F(t),
$$



with the same forcing structure as LTE.

### 2.4 Phase reduction

Let:



$$
A(t) = R(t)e^{i\Phi(t)}.
$$



Under weak forcing:



$$
\frac{d\Phi}{dt} = \sum_i k_i \sin(\omega_i t).
$$



### 2.5 Observable



$$
\eta(t) \sim \sin\Phi(t).
$$



### 2.6 Figure: β-plane modal reduction

![Beta-plane reduction](/assets/media/lte_beta_plane_reduction.svg)

---

## 3. Equivalence of the Two Derivations  
### 3.1 Operator-level mapping

Both derivations reduce to:

- A **linear operator** acting on a state variable.
- A **periodic forcing** term.
- A **single-mode** or **single-coordinate** temporal equation.

### 3.2 Assumption-level mapping

| LTE | β-plane |
|-----|---------|
| Equatorial Coriolis cancellation | Equatorial Hermite mode |
| Scalar PDE | Single-mode amplitude equation |
| Temporal Sturm–Liouville | Meridional Sturm–Liouville |
| Latent phase \(\Phi(t)\) | Latent phase \(\Phi(t)\) |

### 3.3 Latent-variable mapping

Both yield:



$$
\Phi(t) = \sum_i k_i \sin(\omega_i t).
$$



### 3.4 Observable mapping

Both yield:



$$
\zeta(t) = \sin\Phi(t).
$$



### 3.5 Figure: Equivalence map

![Equivalence map](/assets/media/lte_equivalence_map.svg)

---

## 4. Rationalization of the Ansatz  
### 4.1 Why the ansatz is legitimate

The ansatz:



$$
\Phi(t) = \sum_i k_i \sin(\omega_i t)
$$



is not a guess — it is the *general solution* of a forced linear ODE with periodic forcing.

### 4.2 Why it was overlooked

- Traditional GFD emphasizes spatial modes, not temporal latent coordinates.
- The nonlinear observable \(\sin\Phi(t)\) hides the underlying linear structure.
- Climate indices were historically treated as stochastic, not deterministic.

### 4.3 Why it fits modern applied mathematics

- Latent coordinates are standard in reduced-order modeling.
- Nonlinear observables are natural in Koopman theory.
- SINDy identifies exactly this structure when applied to ENSO/QBO/LOD.

---

## 5. Connections to Modern Frameworks  
### 5.1 SINDy

SINDy identifies:

- A **latent coordinate** \(\Phi(t)\)
- A **nonlinear observable** \(\zeta(t)=\sin\Phi(t)\)

### 5.2 Koopman theory

\(\Phi(t)\) is a **Koopman eigenfunction** coordinate.

### 5.3 Topological phase dynamics

The equatorial waveguide acts like a **topological edge state**:

- Bulk rotating fluid = bulk bands  
- Equatorial modes = edge states  
- Slow cyclic forcing = geometric phase  
- \(\Phi(t)\) = topological phase  

### 5.4 Figure: Topological-phase interpretation

![Topological phase diagram](/assets/media/lte_topological_phase.svg)



## QBO results

### Period construction

The GeoEnergyMath QBO analyses report that the QBO period can be constructed from a compact set of lunar timing components, especially the draconic, tropical, and anomalistic months, after annual sampling and aliasing are applied.[^64][^82] The QBO is thereby represented not as an arbitrary emergent oscillation but as a specific aliased product of known lunar cycles.[^64]

This is consistent with the short ESDD paper, which explicitly identified the interaction of lunar nodal timing with the annual cycle as the key to modeling the QBO reversals.[^60] In the author response to reviewers, the doubling from the Chandler wobble timescale to the QBO timescale was further explained as a consequence of nodal-crossing aliasing, analogous in spirit to the semiannual oscillation in the upper equatorial stratosphere but shifted into the lower-frequency band by annual interaction.[^62]

The significance of this construction is that the QBO’s mean 28-month period is not treated as an empirical target to be tuned after the fact.[^60][^64] Instead, it emerges from fixed lunar periods and annual aliasing relations, which gives the model a strong binary quality: if the frequencies are right, the fit persists; if the underlying periods are altered, the alignment degrades quickly.[^64][^82]

### Phase and correlation fits

The archived QBO posts report that direct model fits to the QBO index improve substantially when inclination or nodal modulation is included, with correlation rising from roughly 0.6 in simpler constructions to above 0.8 in the more detailed forcing treatment.[^72] This is one of the most consequential numerical results in the body of work, because it suggests that the finer structure of the lunar forcing is not cosmetic but materially relevant to observed QBO variability.[^72]

The same QBO analyses also emphasize split-training and cross-checking approaches rather than one-pass retrospective fitting.[^77] That is important because the charge most often made against deterministic harmonic models is that they merely overfit a stationary record; the archived training/validation approach is intended to counter exactly that criticism.[^77]

Another noteworthy result in the QBO archive is the treatment of the 2005/2016 anomaly pair, where a modest shift in the annual impulse sequence is reported to restore otherwise degraded alignment and thereby recast the 2015–2016 “disruption” as behavior still compatible with the forcing framework.[^72] Even if one does not accept that interpretation immediately, it is a concrete, testable claim rather than a vague appeal to unseen external influence.[^72]

### Relation to canonical QBO theory

These QBO results do not deny the physical relevance of canonical wave–mean-flow theory.[^9][^73] Standard theory still explains how upward-propagating Kelvin, Rossby-gravity, inertia-gravity, and gravity waves deposit momentum and cause the broad easterly and westerly shear zones to descend over many months.[^5][^9][^12]

What the draconic-forcing interpretation changes is the timing problem.[^60][^64] Rather than assuming that the full phase behavior is emergent from internally generated wave spectra alone, it argues that the internal mechanism may be externally organized or weakly phase-locked by lunisolar timing, with the wave field acting as the medium rather than the ultimate clock.[^36][^41][^60]

That distinction matters because current QBO-capable GCMs display large spreads in simulated period, amplitude, depth, and regularity, indicating that internal-only formulations are sensitive to tuning and structural assumptions.[^6][^73] The absence of a consensus model solution means that the success of an alias-based timing model cannot be dismissed simply because GCMs can also produce QBO-like behavior by other means.[^5][^6]

## Chandler wobble results

The Chandler wobble occupies a special role in this framework because its observed period near 433 days is spectrally sharp and historically important.[^59][^61] The common-mode argument asserts that this period is also recoverable from lunisolar aliasing, specifically through the nodal timing of the Moon interacting with the annual cycle to generate a cyclic torque on the non-spherical rotating Earth.[^59][^60]

The short ESDD paper made this claim explicitly, contrasting it with the longstanding practice of treating the Chandler wobble as a slightly modified free resonance of the Earth system.[^60] The GeoEnergyMath Chandler wobble and related cross-validation posts extend that idea by documenting time-domain fits and validation results using the same forcing family that appears in the QBO and ENSO formulations.[^59][^83][^88]

The Chandler wobble therefore functions as a lemma in the mathematical sense invoked here.[^59][^64] If a compact forcing architecture can explain both the Chandler wobble and the QBO with high spectral precision, then the hypothesis acquires an internal coherence that would be absent if the argument rested on ENSO alone.[^59][^60]

## ENSO and mean sea-level results

The GEM-LTE body of work extends the forcing framework into the equatorial ocean by modeling ENSO and related indices as forced responses of a thermocline-sloping or sloshing system under annual and lunar tidal excitation.[^60][^66] The short ESDD paper summarized this as a Laplace’s tidal equation (LTE) problem in which conventional tidal terms become visible only after appropriate nonlinear and alias-processing treatment.[^60]

What distinguishes the archived GEM-LTE results from a purely spectral argument is their use of validation-style comparisons and their extension across multiple observables.[^66][^83][^86] Publicly posted results report cross-validation on ENSO-like indices and sea-level records, including archived February 2026 evaluations of tide-gauge and MSL behavior under the same forcing framework.[^86]

The significance of those MSL and ENSO fits is twofold.[^60][^86] First, they show that the common-mode forcing can survive the far greater complexity of ocean dynamics. Second, they create a route for physical amplification into the atmosphere, because equatorial ocean variability and thermocline adjustment are known to be dynamically linked to atmospheric Kelvin-wave generation and tropical convection patterns.[^36][^43]

The ESDD author response also underscored that ENSO’s richer spectrum can arise from multiple tidal contributors, including fortnightly and monthly terms whose annual aliases cluster around 3.8–3.9 years and interact through nonlinear processes to create longer-period structure.[^62] This is consistent with the GEM-LTE claim that a common forcing family can explain both the dominant ENSO band and slower amplitude modulation without requiring a separate ad hoc oscillator for each timescale.[^60][^66]

## Common-mode synthesis

The common-mode interpretation is strongest when the three systems are treated together rather than as separate case studies.[^59][^60][^64] The QBO contributes a well-known atmospheric cycle with a mean period near 28 months. The Chandler wobble contributes a rotational cycle near 433 days. ENSO and MSL contribute interannual oceanic variability with broad yet structured spectra.[^59][^60]

The central synthesis is that all three admit forcing descriptions based on the same family of lunisolar periods, the same annual aliasing logic, and the same insistence that the forcing frequencies are fixed by astronomy rather than selected post hoc from the data.[^60][^63][^64] That combination is what gives the argument its common-mode character.[^59][^60]

This is not simply an appeal to “there are lunar signals everywhere.”[^60] The claim is stronger: a single forcing architecture appears repeatedly, at the right frequencies and phases, in systems that are otherwise modeled separately by ocean dynamics, atmospheric dynamics, or rotational mechanics.[^59][^60][^64]

## Kelvin-wave amplification as the novel bridge

The most novel aspect of the present synthesis is the proposed bridge between draconic-forced ocean/sea-level behavior and QBO phase reversals through Kelvin-wave amplification.[^36][^41][^43] This bridge is needed because the QBO is physically realized in the stratosphere through wave-driven momentum deposition, while the common-mode forcing evidence is strongest when all three systems are taken together.[^9][^73]

Observations and theory show that equatorial Kelvin waves and related gravity-wave disturbances can propagate upward through the tropical stratosphere with vertical coherence over several kilometers and periods of a few to roughly 10 days.[^36][^39][^44] These waves can produce nearly in-phase or weak-lag wind fluctuations between levels such as 45 and 65 hPa, not because the mean QBO descends that rapidly, but because the wave structure itself is vertically coherent.[^41][^44]

High-resolution Singapore wind products illustrate exactly this coexistence: broad, slowly descending QBO regimes overlaid by narrow vertical striations that indicate fast, coherent temporal variability.[^22] The broad sloping bands are the familiar QBO, while the striations are compatible with Kelvin-wave and gravity-wave packets acting on daily to weekly timescales.[^9][^36]

The proposed synthesis is that draconic-forced ENSO and MSL behavior can modulate the equatorial wave environment, especially Kelvin-wave activity, which then feeds into the atmospheric momentum budget that controls QBO phase transitions.[^36][^43][^60] In that interpretation, the common-mode forcing does not replace wave–mean-flow interaction; it organizes and amplifies it.[^9][^41]

## Energy-rate plausibility

Back-of-the-envelope energy-rate scaling supports why this bridge is physically plausible.[^36][^41] A Kelvin-wave-like disturbance with order-10 m/s wind amplitude and a period of roughly 10 days cycles kinetic energy at a rate on the order of \(10^{-5}\) W/kg, while a slow QBO mean-flow change of order 20–40 m/s over 14–24 months corresponds to rates on the order of \(10^{-6}\) to \(10^{-5}\) W/kg.[^36][^72]

These estimates do not prove causation, but they show that the fast equatorial wave field operates on a power scale that is not trivially small compared with the slow mean-flow evolution.[^36][^41] That makes it physically credible that externally phased modulation of Kelvin-wave activity could influence when a QBO shear regime reaches reversal threshold, even if the broad descent of the QBO remains a slow process.[^9][^36]

This energy-rate argument also helps reconcile why near-zero or one-day lags can appear between nearby stratospheric levels in filtered high-resolution wind products while the QBO itself still descends over many months.[^22][^41] The fast and slow signals are superposed but dynamically distinct, and the common-mode thesis is specifically about how the fast signal can organize the timing of the slow one.[^36][^60]

## Why existing GCM disagreement matters

A key contextual point made in both the short ESDD paper and subsequent discussion is that there is still no consensus QBO mechanism as realized in comprehensive climate models.[^60][^73] Multi-model evaluations of CMIP-class simulations show large differences in whether a model produces a QBO at all, and if so in its amplitude, depth, period, asymmetry, and regularity.[^6][^73]

This matters because one of the most common objections to a common-mode forcing interpretation is that internal wave–mean-flow dynamics already produce QBO-like behavior in some models.[^5][^73] That statement is true, but incomplete. The same model intercomparison literature also shows that internal-only realizations are not unique and do not converge tightly on a single realistic solution.[^6][^73]

Therefore, the proper comparison standard is not whether a draconic-forcing model looks exactly like one favored GCM.[^6] The proper standard is whether the forcing framework captures observed frequencies, phases, disruptions, and cross-system coherence at least as parsimoniously as existing internal-dynamics explanations.[^60][^73]

## Implications of the ESDD review history

The ESDD submission history matters because it demonstrates that the core argument was placed into the scientific record in a citable, public form even though the venue did not carry it forward as a full accepted paper.[^60][^62] The paper and interactive comments are valuable not only as archival evidence of priority but also because they expose the argument to exactly the sort of criticism that deterministic harmonic models usually face.[^62]

The review exchange also clarified where the strongest misunderstandings arise.[^62] One is the intuitive difficulty of accepting period-doubling from a shorter lunar-derived timescale to the QBO timescale. Another is the tendency to assume that if a phenomenon can be generated internally in a model, then an external timing structure must be irrelevant.[^62][^73]

The archived response addressed both points directly, emphasizing that aliasing can map forcing either to faster or slower bands depending on sampling geometry, and that nonlinear LTE-based dynamics can populate dense power spectra without losing the fingerprint of the original astronomical forcing.[^62] For the purposes of this white paper, the value of that exchange is that it already contains the concise rebuttal to the most immediate objections.[^62]

## Interpretation

Taken together, the archived results support a coherent interpretation.[^59][^60][^64] QBO period structure, Chandler wobble timing, ENSO variability, and MSL behavior can all be described using a common family of draconic and related lunisolar forcings filtered through system-specific dynamics.[^60][^63]

Within that interpretation, the QBO and Chandler wobble are the sharpest spectral demonstrations, while ENSO and MSL provide the broader dynamical validation.[^59][^60][^86] The Kelvin-wave amplification idea then supplies the missing physical bridge linking externally phased ocean variability to the stratospheric momentum-deposition environment in which QBO reversals actually occur.[^36][^41][^43]

The overall picture is therefore neither a purely spectral numerology nor a rejection of established geophysical dynamics.[^9][^60] It is a forcing-plus-filtering framework: fixed astronomical periods provide the timing scaffold, annual aliasing and nonlinear system response move those periods into the observed bands, and oceanic, atmospheric, and rotational dynamics determine the final realized amplitude and waveform.[^60][^62][^64]

## Conclusion

The documented work summarized here reports more than an isolated proposal.[^60][^64] It reports a developed and archived family of analyses in which QBO, ENSO, Chandler wobble, and related sea-level and rotational observables are fit using common draconic-based forcing logic, with publicly available formulas, explanatory posts, repositories, and review exchanges.[^59][^60][^86]

The strongest evidentiary points are the frequency-exact or near-frequency-exact fits for the QBO and Chandler wobble, the cross-validated ENSO/MSL results under the GEM-LTE framework, the improvement of QBO fits when nodal/inclination modulation is added, and the physically plausible Kelvin-wave bridge by which externally phased ocean variability can amplify atmospheric forcing relevant to QBO reversals.[^59][^60][^72][^86]

Whether this framework ultimately becomes accepted, refined, or rejected will depend on how well others can reproduce or falsify the archived fits.[^5][^6] What is already established is that the common-mode draconic interpretation has been laid out in sufficient detail to merit direct technical evaluation rather than dismissal by default appeal to internal variability alone.[^60][^73]

## References

[^5]: Angevine, W. M. et al. (2016). Modeling the QBO—Improvements resulting from higher vertical resolution. Journal of Geophysical Research or related QBO modeling papers.
[^6]: Recent work on QBO period fluctuations and GCM diversity, e.g., “Explaining the period fluctuation of the quasi-biennial oscillation” (Atmospheric Chemistry and Physics, 2025) and related CMIP analyses.
[^9]: Baldwin, M. P., et al. (2001). The quasi-biennial oscillation. Reviews of Geophysics, 39(2), 179–229.
[^12]: Recent studies on improvements to QBO forcing via resolved waves and higher vertical resolution in climate models (AGU / ACP literature circa 2024–2025).
[^22]: Standard observational QBO descriptions and Singapore stratospheric wind climatologies used as canonical QBO references.
[^36]: Randel, W. J., and collaborators (1990s–2000s). Kelvin-wave– induced trace-constituent oscillations and satellite temperature analyses in the tropical stratosphere.
[^39]: Studies documenting Kelvin-wave variability near the equatorial tropopause from reanalysis and satellite data sets.
[^41]: Textbook and lecture-note summaries on equatorial wave theory and Kelvin modes (e.g., “Waves at low latitudes” material in standard atmospheric dynamics courses).
[^43]: Recent reviews of turbulence and Kelvin waves in the tropical stratosphere that discuss wave–mean-flow interactions and intermittency.
[^44]: Studies of Kelvin-wave propagation and vertical coherence in the tropical stratosphere from reanalysis and satellite datasets.
[^50]: Newman, P. A., et al. (2016). The anomalous change in the QBO in 2015–2016. Geophysical Research Letters, 43(16), 8791–8797.
[^52]: Hitchman, M., and coauthors (2021). Observational history of the direct influence of the stratospheric polar vortex and QBO on tropospheric variability.
[^55]: Papers on QBO disruptions and future changes under global warming (e.g., GRL and ACP articles around 2020–2022 on QBO disruption).
[^59]: Pukite, P. (2019). Lunisolar forcing of the Chandler wobble. GeoEnergy Math blog posts and associated analyses tying tidal torques to polar motion.
[^60]: Pukite, P. (2020). Long-period tidal forcing in geophysics — application to ENSO, QBO, and Chandler wobble. Earth System Dynamics Discussions, esd-2020-74.
[^61]: Pukite, P. (2018). Chandler wobble model and forcing analysis. GeoEnergy Math blog post.
[^62]: Pukite, P. (2020). Author response to reviewer comments on esd-2020-74. Earth System Dynamics Discussions interactive comment exchange, clarifying the period-doubling mechanism and ENSO spectrum population by aliased tidal contributors.
[^63]: Pukite, P., Coyne, D., & Challou, D. (2019). Mathematical Geoenergy: Discovery, Depletion, and Renewal. Wiley/AGU Geophysical Monograph 241, especially chapters 11–13 on QBO, ENSO, and Chandler wobble.
[^64]: Pukite, P. (2015). Model of the Quasi-Biennial Oscillation. GeoEnergy Math blog post describing the aliased lunisolar tidal forcing recipe.
[^66]: Pukite, P. (2014–2026). GEM-LTE ENSO and sea-level modeling results and documentation in the GEM-LTE repository and companion posts on GeoEnergy Math (including Feb 2026 cross-validation runs).
[^72]: Pukite, P. (2019). Detailed forcing of QBO. GeoEnergy Math post expanding the draconic modulation and its impact on QBO and ENSO.
[^73]: Idealized and mechanistic QBO modeling work (e.g., Haynes and coauthors) exploring forced QBOs and wave–mean-flow interaction in simplified models.
[^77]: Pukite, P. (2016). Short training intervals and split-sample validation for QBO fits. GeoEnergy Math analysis of robustness to training-window choice.
[^82]: Pukite, P. (2021). QBO aliased harmonics. GeoEnergy Math discussion of how harmonics of the draconic forcing appear under annual aliasing.
[^83]: Pukite, P. (2022). Cross-validation. GeoEnergy Math post formalizing cross-validation as the primary evaluation tool for geophysical models.
[^86]: Pukite, P. (2026). GEM-LTE Feb 2026 experiments: common latent layer across ~100 MSL sites and climate indices, demonstrating cross-validated tidal forcing in sea-level and climate-index data.
[^88]: Pukite, P. (2023). Cross-validation of Chandler wobble and related rotational observables using tidal forcing framework. GeoEnergy Math post.
