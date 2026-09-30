"""Prompt variants adapted from the original retail capstone notebook.

The six analysis prompts preserve the experiment's wording and business context.
"""

def zero_shot_prompt_v1(review_text: str) -> str:
    return f"""Analyze the following customer review and extract/generate the following required fields in JSON format:

    - Category: Such as Fit/Sizing, Quality, Style, Customer Service
    - Sentiment: Positive, Negative, or Neutral
    - Summary: 1 short sentence summarizing the review
    - Personalized Message: A polite customer reply addressing their specific feedback
    - Retail Insight: 1 actionable business insight based on the review

    Review:
    "{review_text}"

    Return ONLY valid JSON matching these keys:
    {{"Category": "", "Sentiment": "", "Summary": "", "Personalized_Message": "", "Retail_Insight": ""}}"""


def zero_shot_prompt_v2(review_text: str) -> str:
    return f"""You are an expert Retail Feedback Analyst at ChicStyle, which is a growing fashion retail platform. Your primary goal is to analyze customer reviews with high precision to improve product offerings and customer satisfaction.

    Task:
    Carefully evaluate the review provided below and extract structured data following these guidelines:

      1. Category: Select the primary category most relevant to the review: Fit/Sizing, Material/Fabric Quality, Pricing, Shipping, Style/Design.
      2. Sentiment: Choose exactly one match to represent the sentiment of the review: Positive, Negative, or Neutral.
      3. Summary: Summarize the main issue or feedback in 20 words or less.
      4. Personalized Message: Draft a professional, empathetic response from ChichStyle addressing the customer directly.
      5. Retail Insight: Provide one concrete, actionable recommendation for inventory, design, or quality assurance that would improve the customer experience.

    Customer Review:
    "{review_text}"

    Output Requirement:
    Return ONLY a valid JSON object with the exact keys: "Category", "Sentiment", "Summary", "Personalized_Message", "Retail_Insight". Do not include markdown formatting outside the JSON block."""


def few_shot_prompt_v1(review_text: str) -> str:
    return f"""Analyze the customer review and extract the fields in JSON format: Category, Sentiment, Summary, Personalized_Message, and Retail_Insight. Use the following examples as guidance for the expected output:

    Example 1:
    Review: "The fabric is incredibly soft and it fits perfectly! Will buy in more colors."
    Output:
    {{"Category": "Fit/Sizing", "Sentiment": "Positive", "Summary": "Customer loves the soft fabric and perfect fit.", "Personalized_Message": "We are thrilled you love the fit and feel of the fabric! We can't wait to see which color you pick next.", "Retail_Insight": "High satisfaction with this fabric; consider using it across other product lines."}}

    Example 2:
    Review: "Zipper broke after one use. Very disappointed for the price I paid."
    Output:
    {{"Category": "Quality", "Sentiment": "Negative", "Summary": "Zipper broke after a single use.", "Personalized_Message": "We are so sorry to hear about the zipper breaking. Please contact our support team for a full refund or replacement.", "Retail_Insight": "Investigate zipper supplier for this batch to prevent future quality assurance failures."}}

    Review: "{review_text}"
    Output:"""


def few_shot_prompt_v2(review_text: str) -> str:
    return f"""You are an expert Retail Feedback Analyst at ChicStyle, which is a growing fashion retail platform. You will analyze customer reviews according to the following rules and examples.

    Rules:
    1. Category: Must be exactly one of: [Fit/Sizing, Quality, Style, Customer Service, Pricing].
    2. Sentiment: Must be exactly one of: [Positive, Negative, Neutral].
    3. Summary: Must be strictly under 15 words.
    4. Personalized_Message: Must address the customer politely and offer a solution if negative.
    5. Retail_Insight: Must provide a specific action for the business.

    Example 1:
    Review: "The fabric is incredibly soft and it fits perfectly! Will buy in more colors."
    Output:
    {{"Category": "Fit/Sizing", "Sentiment": "Positive", "Summary": "Customer praises the fabric softness and fit.", "Personalized_Message": "We are thrilled you love the fit and feel of the fabric! We can't wait to see which color you pick next.", "Retail_Insight": "High satisfaction with this fabric; consider using it across other product lines."}}

    Example 2:
    Review: "Zipper broke after one use. Very disappointed for the price I paid."
    Output:
    {{"Category": "Quality", "Sentiment": "Negative", "Summary": "Zipper broke after a single use.", "Personalized_Message": "We are so sorry to hear about the zipper breaking. Please contact our support team for a full refund or replacement.", "Retail_Insight": "Investigate zipper supplier for this batch to prevent future quality assurance failures."}}

    Carefully analyze the following review using the rules above:
    Review: "{review_text}"
    Output (Return ONLY valid JSON):"""


def cot_prompt_v1(review_text: str) -> str:
    return f"""Analyze the customer review below. Think step-by-step to identify the key points, customer emotions, and operational implications before generating the final JSON output.

Customer Review:
"{review_text}"

Task:
1. Reason step-by-step about what the customer is saying.
2. Based on your reasoning, output ONLY a valid JSON object with these exact keys:
   {{"Category": "", "Sentiment": "", "Summary": "", "Personalized_Message": "", "Retail_Insight": ""}}"""


def cot_prompt_v2(review_text: str) -> str:
    return f"""You are an expert Retail Feedback Analyst at ChicStyle, which is a growing fashion retail platform.

Follow this systematic reasoning process internally:
- Step 1: Identify the main issue or highlight mentioned (e.g., sizing, defect, delivery delay).
- Step 2: Determine the customer's emotional tone and overall sentiment (Positive, Negative, or Neutral).
- Step 3: Determine the single most applicable category from: [Fit/Sizing, Quality, Style, Customer Service, Pricing].
- Step 4: Formulate a concise summary under 15 words.
- Step 5: Draft an empathetic customer response resolving their specific problem.
- Step 6: Identify one strategic, high-impact business recommendation for operations or QA.

Customer Review:
"{review_text}"

Return ONLY a valid JSON object. Do not include markdown formatting outside the JSON block:
{{"Category": "", "Sentiment": "", "Summary": "", "Personalized_Message": "", "Retail_Insight": ""}}"""


def recommendation_prompt(review_text: str) -> str:
    return f"""You are a product recommendation classifier for Luxe Apparel Co.

Task:
Analyze the customer review below and predict whether the customer recommends the product.

Rules:
1. Recommended_IND: Output strictly 1 if the customer recommends/likes the product, or 0 if they do not.
2. Reason: Provide a 1-sentence justification directly referencing the review.

Customer Review:
"{review_text}"

Return ONLY valid JSON in this exact structure:
{{"Recommended_IND": 1, "Reason": "The customer was satisfied with the fit and quality."}}"""


ANALYSIS_PROMPTS = {
    "zero_v1": zero_shot_prompt_v1,
    "zero_v2": zero_shot_prompt_v2,
    "few_v1": few_shot_prompt_v1,
    "few_v2": few_shot_prompt_v2,
    "cot_v1": cot_prompt_v1,
    "cot_v2": cot_prompt_v2,
}

