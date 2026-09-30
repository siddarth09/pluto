# Review papers

The PDFs are gitignored: 162 MB, and all of them are public on arXiv. Re-fetch with

```bash
cd ~/projects25/src/pluto/mimic/review
while read -r slug id _; do
  [ -z "$slug" ] && continue
  curl -sL -A "Mozilla/5.0" "https://arxiv.org/pdf/$id" -o "$slug.pdf"
done < <(grep -E '^[0-9]' MANIFEST.md | awk '{print $1, $2}')
```

| file | arXiv | title |
|---|---|---|
| 01_deepmimic | 1804.02717 | DeepMimic: Example-Guided Deep RL of Physics-Based Character Skills |
| 02_amp | 2104.02180 | AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control |
| 03_ase | 2205.01906 | ASE: Large-Scale Reusable Adversarial Skill Embeddings |
| 04_beyondmimic | 2508.08241 | BeyondMimic: From Motion Tracking to Versatile Humanoid Control |
| 05_what_matters_gmt | 2607.19903 | What Matters in Humanoid General Motion Tracking? |
| 06_phuma | 2510.26236 | PHUMA: Physically Reliable Humanoid Locomotion Dataset |
| 07_exbody | 2402.16796 | Expressive Whole-Body Control for Humanoid Robots |
| 08_exbody2 | 2412.13196 | ExBody2: Advanced Expressive Humanoid Whole-Body Control |
| 09_h2o | 2403.04436 | Learning Human-to-Humanoid Real-Time Whole-Body Teleoperation |
| 10_omnih2o | 2406.08858 | OmniH2O: Universal and Dexterous Human-to-Humanoid Teleoperation |
| 11_humanplus | 2406.10454 | HumanPlus: Humanoid Shadowing and Imitation from Humans |
| 12_phc | 2305.06456 | Perpetual Humanoid Control for Real-time Simulated Avatars |
| 13_sonic | 2511.07820 | SONIC: Supersizing Motion Tracking for Natural Humanoid Control |
| 14_mjlab | 2601.22074 | mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning |
| 15_limmt | 2606.06953 | LIMMT: Less is More for Motion Tracking |

Reading order is in `../README.md`. Start with 01, then 04.
