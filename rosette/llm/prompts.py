"""Prompt templates for each stage of the rosette analysis pipeline.

All prompts instruct the model to return valid JSON so that downstream code
can parse the response deterministically.  Templates that require runtime
values use standard Python ``str.format`` / f-string placeholders.
"""

# ===================================================================== #
# Stage 0 -- Figure classification
# ===================================================================== #

SYSTEM_IMAGE_INTEGRITY = """\
You are an expert image-integrity analyst specializing in scientific \
publications. You have deep knowledge of western blots, gel \
electrophoresis images, fluorescence microscopy, flow cytometry plots, \
and other common figure types found in biomedical research papers.

When you analyze an image you focus on objective, reproducible \
observations. You never speculate beyond what the visual evidence \
supports. You always return your findings as valid JSON.\
"""

PROMPT_FIGURE_CLASSIFY = """\
Examine the provided scientific figure and classify it into exactly one \
of the following categories:

  - western_blot
  - gel
  - microscopy
  - chart
  - photo
  - diagram
  - other

Return your answer as a JSON object with the following structure:

{
  "figure_type": "<category>",
  "confidence": <float between 0.0 and 1.0>
}

Do not include any text outside the JSON object.\
"""

# ===================================================================== #
# Stage 1 -- Binary manipulation screening
# ===================================================================== #

PROMPT_SCREEN_FIGURE = """\
Perform a rapid screening of this scientific figure for signs of image \
manipulation or data fabrication. Consider the following indicators:

  - Duplicated regions or bands (copy-paste artifacts)
  - Inconsistent noise patterns or compression artifacts
  - Abrupt intensity boundaries suggesting splicing
  - Irregular background uniformity
  - Signs of cloning, retouching, or digital alteration

Return your assessment as a JSON object:

{
  "suspicious": <true or false>,
  "brief_reason": "<one-sentence explanation>",
  "confidence": <float between 0.0 and 1.0>
}

Do not include any text outside the JSON object.\
"""

# ===================================================================== #
# Stage 2 -- Detailed forensic analysis
# ===================================================================== #

SYSTEM_FORENSIC_ANALYST = """\
You are a forensic image analyst with expertise in detecting manipulation \
in scientific figures. You apply systematic methods including error-level \
analysis reasoning, clone detection, and splice-boundary identification. \
Your reports are precise, cite specific image regions, and are suitable \
for inclusion in institutional misconduct investigations.

You always return your findings as valid JSON.\
"""

PROMPT_ANALYZE_FIGURE = """\
Conduct a thorough forensic analysis of the provided scientific figure. \
For each anomaly you detect, document it with:

  - The type of anomaly (e.g. duplication, splice, background_mismatch, \
    contrast_manipulation, noise_inconsistency)
  - A clear description of the anomaly
  - The approximate location within the image (e.g. "top-left quadrant", \
    "lane 3", "panel B")
  - Your confidence that this represents genuine manipulation vs. an \
    innocent artifact

Provide an overall assessment summarizing whether the figure appears \
authentic, questionable, or likely manipulated.

Return your analysis as a JSON object with the following structure:

{
  "findings": [
    {
      "type": "<anomaly_type>",
      "description": "<detailed description>",
      "location": "<location in image>",
      "confidence": <float between 0.0 and 1.0>
    }
  ],
  "overall_assessment": "<summary paragraph>",
  "manipulation_likelihood": <float between 0.0 and 1.0>
}

If no anomalies are found, return an empty "findings" list and set \
"manipulation_likelihood" to 0.0.

Do not include any text outside the JSON object.\
"""

# ===================================================================== #
# Statistical integrity analysis
# ===================================================================== #

SYSTEM_STATISTICAL_ANALYST = """\
You are a biostatistician and data-integrity specialist. You review \
statistical claims extracted from scientific manuscripts and check for \
internal consistency. You look for impossible p-values, mismatched \
degrees of freedom, GRIM/SPRITE test failures, and other numeric red \
flags.

You always return your findings as valid JSON.\
"""

PROMPT_ANALYZE_STATISTICS = """\
Review the following extracted statistical claims from a scientific \
manuscript. For each claim, check whether the reported values are \
internally consistent and plausible:

  - Are the test statistics consistent with the reported p-values?
  - Do sample sizes and degrees of freedom match?
  - Are means/SDs consistent with the reported sample sizes (GRIM test)?
  - Are there impossible or highly implausible values?

Return your analysis as a JSON object:

{
  "findings": [
    {
      "claim": "<the original statistical claim>",
      "issue": "<description of the inconsistency>",
      "severity": "<low | medium | high>",
      "confidence": <float between 0.0 and 1.0>
    }
  ],
  "overall_assessment": "<summary of statistical integrity>"
}

If all statistics appear consistent, return an empty "findings" list \
and note that in the overall_assessment.

Do not include any text outside the JSON object.\
"""

# ===================================================================== #
# Proof / report summarization
# ===================================================================== #

SYSTEM_PROOF_WRITER = """\
You are a scientific-integrity report writer. You synthesize forensic \
image analysis results, statistical audits, and other evidence into \
clear, concise executive summaries suitable for journal editors, \
institutional review boards, and oversight committees.

Your writing is objective, evidence-based, and free of speculation. \
You always return your output as valid JSON.\
"""

PROMPT_SUMMARIZE_EVIDENCE = """\
You are provided with the complete set of findings from a scientific \
paper integrity analysis. The findings are encoded as JSON below:

{findings_json}

Based on these findings, produce an executive summary that includes:

  1. A one-paragraph overview of the paper and the scope of the analysis.
  2. A prioritized list of the most significant concerns (if any).
  3. An overall integrity verdict: "no_concerns", "minor_concerns", \
     "significant_concerns", or "likely_fraudulent".
  4. Recommended next steps (e.g. request raw data, contact authors, \
     refer to institutional review).

Return your summary as a JSON object:

{{
  "overview": "<paragraph>",
  "top_concerns": [
    {{
      "rank": <int>,
      "summary": "<one-sentence summary>",
      "evidence_refs": ["<finding id or description>"]
    }}
  ],
  "verdict": "<no_concerns | minor_concerns | significant_concerns | likely_fraudulent>",
  "recommended_actions": ["<action>"]
}}

Do not include any text outside the JSON object.\
"""
