# RealClimate `@whut` archive: argument categories in Pukite's model proposals

This note is based on the **exact-author** corpus in `output/realclimate_whut_comments.json` and `output/realclimate_whut_comments.md`, not the earlier over-collected archive. The current corpus contains **931 comments** from 2016-2026, almost all under `Paul Pukite (@whut)`, with a few `Paul Pukite` / `Paul Pukite (@WHUT)` variants.

All comment references below point into the local archive: `output/realclimate_whut_comments.md`.

## Short version

The recurring model package is:

1. **ENSO, QBO, and related indices are not primarily chaotic autonomous oscillators.**
2. **They are better treated as forced, non-autonomous responses to lunisolar/tidal inputs.**
3. **The observable patterns are standing-wave or sloshing behaviors constrained by equatorial geometry and boundary conditions.**
4. **The right way to test these models is historical cross-validation, not just waiting years for forward predictions.**
5. **The same forcing logic may extend to mean sea level, AMO, delta-LOD, Chandler wobble, and other geophysical indices.**

## 1. Forced-response view of ENSO and QBO

This is the most persistent claim in the archive. Pukite repeatedly argues that ENSO and QBO should not be modeled as self-excited internal oscillators that spontaneously emerge from a chaotic background. Instead, they should be treated more like tides: externally forced, phase-sensitive responses.

### Core claims

- ENSO and QBO are **non-autonomous** systems.
- Their apparent irregularity reflects complicated forcing and response, not irreducible chaos.
- Mainstream GCM treatments too often assume autonomous oscillation where explicit forcing should be included.

### Representative comments

- [Comment 652930](output/realclimate_whut_comments.md#comment-652930) says QBO is likely forced by seasonally aliased monthly tidal cycles and argues that both QBO and ENSO need explicit forcing inputs.
- [Comment 660487](output/realclimate_whut_comments.md#comment-660487) argues that ENSO has deterministic properties and cites signal-processing work against the "completely unpredictable" framing.
- [Comment 682647](output/realclimate_whut_comments.md#comment-682647) explicitly compares ENSO to ocean tides and says boundary conditions and forcing matter more than butterfly-effect sensitivity.
- [Comment 686879](output/realclimate_whut_comments.md#comment-686879) says ENSO should be tested as a deterministic process by feeding in lunisolar forcing directly.

### Interpretation

The central modeling move is to replace **internal variability first** with **forced response first**. That is the conceptual bridge connecting the RealClimate comments to the later LTE / reduced-order modeling work elsewhere in this repo.

## 2. Lunisolar and tidal forcing as the candidate driver

Pukite's preferred forcing is not vague "natural variability." It is much more specific: **lunisolar gravitational/tidal forcing**, often acting through thermocline motion, subsurface waves, or seasonally aliased harmonics.

### Core claims

- QBO can be explained by seasonally aliased lunar tidal forcing.
- ENSO switching is linked to subsurface waves and tidal forcing.
- Lunar/solar cycles are measurable in other geophysical observables such as delta-LOD.
- The thermocline is the especially sensitive part of the ocean-atmosphere system.

### Representative comments

- [Comment 660635](output/realclimate_whut_comments.md#comment-660635) lays out the QBO case most explicitly: wind acceleration, latitudinal displacement tied to tidal elevation, and seasonal aliasing of lunar tractive forces to match the observed QBO period.
- [Comment 680663](output/realclimate_whut_comments.md#comment-680663) says lunisolar gravitational forcing is emerging as a mechanism for ENSO and other flow patterns.
- [Comment 721359](output/realclimate_whut_comments.md#comment-721359) cites JPL work on lunar tidal forcing in ocean winds, rainfall, OLR, wave height, and tropical instability waves.
- [Comment 760628](output/realclimate_whut_comments.md#comment-760628) argues that delta-LOD shows a faithful lunar-cycle response, including the 18.6-year nodal envelope.
- [Comment 788212](output/realclimate_whut_comments.md#comment-788212) ties AMO/rotation-rate variations to the 14.765-day Msf tidal factor and quotes a Perigaud/JPL line of argument.

### Interpretation

This is not a generic "the Moon matters somehow" position. The comments repeatedly narrow the mechanism to **specific harmonic tidal factors**, **aliasing**, **subsurface thermocline response**, and **angular-momentum exchange**.

## 3. ENSO and QBO as standing-wave / sloshing / waveguide phenomena

Another strong theme is that these climate behaviors should be understood geometrically. Pukite often frames ENSO as a Pacific standing wave or sloshing thermocline dipole, and QBO as a highly symmetric equatorial behavior with fewer degrees of freedom than people assume.

### Core claims

- ENSO is a **standing-wave** or **sloshing dipole** tied to Pacific boundary conditions.
- QBO is simpler than ENSO because it has stronger symmetry and fewer incommensurate forcings.
- Equatorial structure can insulate these modes from turbulence.
- Hemispheric asymmetry and nodal crossings matter for how forcing appears.

### Representative comments

- [Comment 652930](output/realclimate_whut_comments.md#comment-652930) calls ENSO a single standing-wave behavior over the equatorial Pacific.
- [Comment 686879](output/realclimate_whut_comments.md#comment-686879) describes ENSO as a sloshing dipole in the equatorial Pacific thermocline.
- [Comment 704565](output/realclimate_whut_comments.md#comment-704565) argues that ENSO and QBO are standing waves protected by equatorial/topological structure rather than dominated by turbulence.
- [Comment 820565](output/realclimate_whut_comments.md#comment-820565) says an idealized QBO can be captured by a mathematical construction with the right topology, then mapped to empirical observations.

### Interpretation

This is where the comments most clearly anticipate the repo's later emphasis on equatorial reduction, standing-wave structure, and low-order forced dynamics.

## 4. Model testing: machine learning, curve fitting, and cross-validation

Pukite is not arguing for a black-box model. The methodological preference is closer to: use machine learning or signal processing to discover candidate forcings, then validate reduced physical models by cross-validation on long time series.

### Core claims

- Machine learning is useful for discovering forcing terms or dead ends, not as a substitute for mechanism.
- Curve fitting is acceptable when it extracts physical parameters rather than merely interpolating.
- Historical **out-of-band cross-validation** is the right standard in Earth science where controlled experiments are impossible.
- Overfitting is a risk, but the answer is stronger cross-validation, not abandoning reduced models.

### Representative comments

- [Comment 652930](output/realclimate_whut_comments.md#comment-652930) presents machine learning and data mining as tools for discovering QBO/ENSO forcings.
- [Comment 743109](output/realclimate_whut_comments.md#comment-743109) describes cross-validation of a fluid-dynamics-based model of cyclic climate indices.
- [Comment 788923](output/realclimate_whut_comments.md#comment-788923) says there is already enough data for detailed cross-validation and that the bottleneck is methodological adoption.
- [Comment 792784](output/realclimate_whut_comments.md#comment-792784) describes train-on-one-interval, test-on-out-of-band intervals as the practical antidote to overfitting.
- [Comment 793206](output/realclimate_whut_comments.md#comment-793206) argues that for slow oscillations like ENSO, validation by future prediction alone is too slow and must be supplemented by intensive cross-validation.

### Interpretation

This is important because the archive does **not** just say "fit cycles." It says: use structure-aware fits, discover forcing harmonics, and validate rigorously on held-out intervals.

## 5. Extension to other observables: mean sea level, AMO, delta-LOD, Chandler wobble

By the later comments, the same framework is being extended far beyond ENSO and QBO.

### Core claims

- Mean sea level residuals may encode tidal/common-mode forcing plus links to ENSO and NAO.
- AMO may contain a tidal component.
- Earth rotation and Chandler wobble provide cleaner geophysical evidence for torque/forcing.
- A unified forcing picture may connect multiple geophysical indices.

### Representative comments

- [Comment 783391](output/realclimate_whut_comments.md#comment-783391) says Chandler wobble should be viewed as a forced response to lunar and solar nodal cycles.
- [Comment 822693](output/realclimate_whut_comments.md#comment-822693) discusses mean sea level as a 19-year averaging problem because of tidal phase and amplitude cycles.
- [Comment 837451](output/realclimate_whut_comments.md#comment-837451) describes cross-validating residual mean-sea-level oscillations at dozens of long-record ports.
- [Comment 839465](output/realclimate_whut_comments.md#comment-839465) argues that multi-year sea-level oscillations and ocean indices can be unified by a common tidal-forcing mechanism.
- [Comment 840971](output/realclimate_whut_comments.md#comment-840971) links internal waves, ENSO, sea-level variation, and NAO-correlated Baltic sea-level cycles in one forcing chain.

### Interpretation

The later comments broaden the proposal from "tidal forcing might matter for ENSO/QBO" to "a common forcing vocabulary may explain multiple coupled geophysical observables."

## What the archive consistently opposes

Across the corpus, Pukite is usually arguing against four opposing ideas:

1. **ENSO/QBO are inherently chaotic and not worth deterministic modeling.**
2. **Wind anomalies are the primary initiator rather than a downstream response.**
3. **Tidal/lunar inputs are too weak or too slow to matter at interannual scales.**
4. **Validation should rely mainly on future prediction rather than historical cross-validation.**

## Bottom line

The corrected RealClimate corpus supports a much sharper summary than the earlier mistaken archive:

1. **Pukite's core proposal is a forced-response model of interannual climate variability.**
2. **The preferred forcing source is lunisolar/tidal, often acting through subsurface ocean dynamics.**
3. **ENSO and QBO are treated as deterministic standing-wave or sloshing responses, not primarily chaotic autonomous oscillators.**
4. **Cross-validation on long historical time series is his preferred validation standard.**
5. **The same framework is repeatedly extended to sea level, AMO, delta-LOD, and Chandler wobble.**

## Bibliography of important links

These are the external links that most clearly define the mature version of the model package described in the comments.

| Link | Why it matters |
| --- | --- |
| https://www.nature.com/articles/s41598-019-49678-w | The most-cited third-party support link in the archive: subsurface ocean waves and likely lunar tidal forcing for ENSO switching. |
| https://agupubs.onlinelibrary.wiley.com/doi/10.1002/9781119434351.ch11 | Pukite's QBO chapter in *Mathematical Geoenergy*; cited as the long-form model for QBO. |
| https://agupubs.onlinelibrary.wiley.com/doi/10.1002/9781119434351.ch12 | Pukite's ENSO chapter in *Mathematical Geoenergy*; cited as the long-form model for ENSO. |
| https://esd.copernicus.org/preprints/esd-2020-74/ | "Unforced Variations should be a Forced Response"; used in comments on thermocline sensitivity and Chandler wobble forcing. |
| https://geoenergymath.com/2024/03/25/proof-for-allowed-modes-of-an-ideal-qbo/ | Mature statement of the topological / allowed-modes argument for an idealized QBO. |
| https://geoenergymath.com/2022/01/14/sea-level-height-as-a-proxy-for-enso/ | Repeatedly cited for using sea-level height as an ENSO proxy and for tying sea-level variability back to climate indices. |
| https://pukite.substack.com/p/mean-sea-level-models | Used for the expanded mean-sea-level modeling program and the claim that tidal forcing synchronizes multiple observables. |
| https://geoenergymath.com/2024/09/23/amo-and-the-mt-tide/ | Representative later link extending the same framework to AMO. |
| https://geoenergymath.com/2024/11/10/lunar-torque-controls-all/ | Representative late-stage synthesis link for the broad torque/forcing view. |
| https://github.com/pukpr/GEM-LTE | Code/data repository linked in later comments for the broader LTE-style modeling framework. |

## Representative comment anchors

If you want to re-read the archive in a high-signal order, start here:

- [Comment 652930](output/realclimate_whut_comments.md#comment-652930) — QBO tidal forcing, ENSO biennial forcing, non-autonomous systems, ML
- [Comment 660635](output/realclimate_whut_comments.md#comment-660635) — compact QBO derivation and seasonal aliasing of lunar tractive forces
- [Comment 682647](output/realclimate_whut_comments.md#comment-682647) — Hawkmoth vs Butterfly framing; ENSO as boundary-forced like tides
- [Comment 686879](output/realclimate_whut_comments.md#comment-686879) — ENSO as deterministic sloshing thermocline dipole
- [Comment 743109](output/realclimate_whut_comments.md#comment-743109) — cross-validation of cyclic climate-index models
- [Comment 760628](output/realclimate_whut_comments.md#comment-760628) — delta-LOD and the 18.6-year nodal envelope
- [Comment 783391](output/realclimate_whut_comments.md#comment-783391) — Chandler wobble as a forced response
- [Comment 820565](output/realclimate_whut_comments.md#comment-820565) — allowed modes / topology argument for ideal QBO
- [Comment 839465](output/realclimate_whut_comments.md#comment-839465) — mean sea level, tidal factors, and common-mode forcing
