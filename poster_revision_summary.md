# Poster Revision Summary

## Calibration Changes
- Revised the poster narrative from "vulnerability demonstrated" to a mixed-result conclusion:
  behavior poisoning sometimes produces positive matched-seed degradation, but reliable degradation is not shown.
- Replaced the key finding banner with cautious language: effects are mixed, seed-sensitive, and non-monotonic in `p`.
- Added an explicit answer/caution box: the research question is answered "sometimes, but not reliably."
- Replaced the core idea flow so it ends in "seed-dependent clean-evaluation outcome" rather than guaranteed degradation.

## Section Updates
- Renamed the main result section to "Mixed Effects, Not Reliable Degradation."
- Rewrote main result captions and interpretation bullets to distinguish `D > 0` degradation from `D < 0` improvement.
- Rewrote the seed sensitivity caption to emphasize the red/blue mixture and the absence of stable degradation.
- Rewrote the mechanism caption to say coverage/collision plots do not reveal a consistent degradation mechanism.
- Reframed the defense box as future defense ideas, not required defenses for a proven attack.
- Rewrote the hypothesis table so reliable degradation, targeted dominance, and monotonic harm are not supported.
- Rewrote takeaways and the final lesson to emphasize intermittent effects and exploratory conclusions.

## Figure Changes
- Regenerated the main degradation plot from the existing summary CSVs after changing its title to avoid "monotonic damage" wording.
- Regenerated the heatmap after changing the colorbar label from "more harm" to "more degradation."
- No numerical values were changed.

## Data and Claims
- No new experiments were run.
- The main reported means were preserved:
  p=0.1: Random 9.43, Targeted no-op -0.21, KL-targeted 4.58.
  p=0.2: Random 0.65, Targeted no-op 1.66, KL-targeted 3.39.
  p=0.3: Random -7.75, Targeted no-op -7.06, KL-targeted 2.47.
- The conclusion is now calibrated: this simple_spread sweep supports seed sensitivity, non-monotonicity, and intermittent degradation, not reliable attack success.

## Verification
- Searched the source for overstrong terms such as vulnerability, harmful, damage, hurt, and permanent.
- Compiled locally with MiKTeX `pdflatex`.
- Rasterized the final PDF preview with `pdftoppm` and checked for clipping.
- Checked the PDF text layer with `pdftotext`; the title appears once.
- Verified the exported PDF is one page with `pdfinfo`.
