"""Specialized LLM prompt templates for western blot analysis.

Purpose-built prompts for detecting duplicated bands, splice boundaries,
identical loading controls, and inappropriate contrast adjustments in
western blot images.
"""

SYSTEM_WESTERN_BLOT_ANALYST = """\
You are a specialist in western blot image analysis with extensive experience \
in detecting image manipulation in biomedical publications. You understand \
western blot artifacts, lane loading patterns, antibody specificity, and \
the technical limitations of film-based and digital western blot imaging \
systems.

You focus on objective, reproducible observations. You distinguish between \
genuine artifacts (uneven transfer, air bubbles, overexposure) and signs \
of intentional manipulation (band duplication, lane splicing, background \
patching, contrast manipulation). You always return your findings as valid \
JSON.\
"""

PROMPT_WESTERN_BLOT_SCREEN = """\
Examine this western blot image and screen for signs of manipulation. \
Focus on these specific indicators:

1. **Duplicated bands**: Look for bands that appear identical across \
   different lanes (same shape, intensity profile, and noise pattern).
2. **Splice boundaries**: Check for vertical discontinuities in the \
   background between lanes that suggest lanes were spliced from \
   different blots.
3. **Background inconsistencies**: Look for rectangular patches of \
   different background intensity that suggest selective editing.
4. **Loading control anomalies**: Check if the loading control (e.g. \
   actin, GAPDH, tubulin) shows suspiciously uniform bands across all \
   lanes.
5. **Contrast manipulation**: Look for lanes with different contrast/\
   brightness levels that suggest selective enhancement.

Return your assessment as a JSON object:

{
  "suspicious": <true or false>,
  "indicators": [
    {
      "type": "<duplicated_bands | splice_boundary | background_patch | loading_anomaly | contrast_manipulation>",
      "description": "<specific description of what you observe>",
      "location": "<location in the image, e.g. 'between lanes 3 and 4'>",
      "confidence": <float between 0.0 and 1.0>
    }
  ],
  "overall_assessment": "<summary of findings>",
  "manipulation_likelihood": <float between 0.0 and 1.0>
}

Do not include any text outside the JSON object.\
"""

PROMPT_WESTERN_BLOT_DETAILED = """\
Perform a detailed forensic analysis of this western blot image. For each \
panel or section visible:

1. **Lane-by-lane comparison**: Compare band patterns across all lanes. \
   Note any lanes that share suspiciously similar band shapes or noise \
   patterns.
2. **Background analysis**: Examine the background between and around \
   lanes for discontinuities, rectangular patches, or gradients that \
   abruptly change direction.
3. **Band intensity profiling**: Note whether band intensities follow a \
   plausible biological pattern (e.g. dose-response, time course) or \
   whether they appear artificially uniform or discontinuous.
4. **Splice detection**: Look for vertical lines or seams between lanes \
   where the background texture, noise level, or gradient direction \
   changes abruptly.
5. **Reuse detection**: Check if any bands appear to be exact copies of \
   other bands (same pixel-level detail), possibly flipped or rotated.

Return your analysis as a JSON object:

{{
  "findings": [
    {{
      "type": "<finding_type>",
      "description": "<detailed description>",
      "location": "<specific location>",
      "affected_lanes": [<list of lane numbers>],
      "confidence": <float between 0.0 and 1.0>
    }}
  ],
  "lane_assessment": [
    {{
      "lane": <lane number>,
      "appears_authentic": <true or false>,
      "notes": "<any observations about this lane>"
    }}
  ],
  "overall_assessment": "<summary paragraph>",
  "manipulation_likelihood": <float between 0.0 and 1.0>
}}

Do not include any text outside the JSON object.\
"""
