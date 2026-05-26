"""
Prompt Templates for Causal Extraction Task
Includes 6 specialized prompts for biomedical causal extraction
"""

import json
from typing import Dict, List


class ExtractionPrompts:
    """Prompt templates for causal extraction with domain-specific instructions"""
    
    # Base instructions used across all prompts
    BASE_INSTRUCTIONS = """Role: You are an expert causal relation extraction system. Your task is to perform Pairwise Causal Discovery (PCD) on the provided text.

Objective: Extract all causal relations as tuples consisting of {Cause, Effect, Causality_Type, Sententiality}.

1. Definition of Causality:
Principle: Causation is the "highest form of association." You must distinguish between factual causal assertions and reported claims/associations.
Distinguish strictly between Cause (the agent/trigger) and Effect (the outcome).
"Can cause" is accepted because it shows a causal relationship even if the relationship might not be the case in that sentence.
"May cause, could cause" and other forms of hedging are not causal facts.
Exclusion: Do not extract relations based on "reported claims" (e.g., "Scientists believe X causes Y" should be ignored).
It is important to annotate causality based on the semantic structure of the text, treating the text independently, with only common sense interpretation. Attempts to decipher abbreviations, or use external definitions of terms used in the text to contextualise relationships are not allowed.

2. Axis 1: Marker Presence (Causality Type):
Classify the relation based on lexical cues:
Explicit (M=1): The text contains clear cue words indicating directionality. For the purpose of this experiments, those words are all variations of: "due to," "results in" "leads to," "causes", "effect of").
Implicit (M=0): Any other instance of causality that does not include variations of the cue words listed above. The relation is inferred from semantic ordering or adjectival phrasing without a verb marker. Example: "X increased Y", "X brought about Y", or "Y attenuated X."

Axis 2: Textual Scope (Sententiality):
Intra-sentential (B=0): The Cause and Effect entities are located within the same sentence.
Inter-sentential (B=1): The Cause and Effect are located in different sentences. You must bridge the logic across sentence boundaries to capture the flow of scientific argumentation.

Axis 3: Cardinality (r):
Decomposition: You must decompose complex sentences into distinct pairwise relations.
Input: "A causes B and C." Output: Pair 1: (A ? B), Pair 2: (A ? C).
Or Input: "A and B can cause C." Output: Pair 1: (A ? C), Pair 2: (B ? C).
It is important to decipher when A independently cause C and B also independently cause C versus when A and B combine to cause C, in the later case, you should not decompose the pairing.

5. Do NOT infer transitive closure. If the text says "A causes B, which causes C," extract (A ? B) and (B ? C). Do not extract (A ? C) unless the text explicitly states "A causes C."

Passive Voice: Canonicalize passive constructions to active forms (e.g., "Y is caused by X" ? Cause: X, Effect: Y)."""
    
    JSON_OUTPUT_FORMAT = """
IMPORTANT: Provide your final output in JSON format:
{
  "pairs": [
    {
      "cause": "identified cause text",
      "effect": "identified effect text",
      "causality": "Explicit or Implicit",
      "sententiality": "Intra or Inter"
    }
  ]
}

Mapping rules:
- M=1 (Explicit markers present) ? "causality": "Explicit"
- M=0 (No explicit markers) ? "causality": "Implicit"
- B=0 (Same sentence) ? "sententiality": "Intra"
- B=1 (Different sentences) ? "sententiality": "Inter"

If no causal relationships exist, return {"pairs": []}"""
    
    @staticmethod
    def zero_shot(sentence: str) -> str:
        """Zero-shot prompt - Direct extraction"""
        return f"""{ExtractionPrompts.BASE_INSTRUCTIONS}

Input Text:
"{sentence}"

{ExtractionPrompts.JSON_OUTPUT_FORMAT}"""
    
    @staticmethod
    def few_shot(sentence: str) -> str:
        """Few-shot prompt with examples"""
        return f"""{ExtractionPrompts.BASE_INSTRUCTIONS}

Examples:

Input: "Hypertension leads to heart failure and kidney damage."
Output:
{{
  "pairs": [
    {{
      "cause": "Hypertension",
      "effect": "heart failure",
      "causality": "Explicit",
      "sententiality": "Intra"
    }},
    {{
      "cause": "Hypertension",
      "effect": "kidney damage",
      "causality": "Explicit",
      "sententiality": "Intra"
    }}
  ]
}}

Input: "Researchers suggest that excessive sugar intake may cause metabolic syndrome."
Output:
{{
  "pairs": []
}}
(Reason: "suggest that... may cause" falls under reported claims and hedging)

Input: "Elevated cortisol levels attenuated immune response. This suppression increased susceptibility to infection."
Output:
{{
  "pairs": [
    {{
      "cause": "Elevated cortisol levels",
      "effect": "immune response attenuation",
      "causality": "Implicit",
      "sententiality": "Intra"
    }},
    {{
      "cause": "The suppression of immune response attenuation",
      "effect": "increased susceptibility to infection",
      "causality": "Implicit",
      "sententiality": "Inter"
    }}
  ]
}}

Input: "The combination of alcohol and sedatives can cause respiratory depression."
Output:
{{
  "pairs": [
    {{
      "cause": "combination of alcohol and sedatives",
      "effect": "respiratory depression",
      "causality": "Explicit",
      "sententiality": "Intra"
    }}
  ]
}}

Task:
Input Text: "{sentence}"

{ExtractionPrompts.JSON_OUTPUT_FORMAT}"""
    
    @staticmethod
    def cot(sentence: str) -> str:
        """Chain of Thought (CoT) prompt"""
        return f"""{ExtractionPrompts.BASE_INSTRUCTIONS}

Instructions:
For every potential relationship in the text, you must perform a step-by-step reasoning process before outputting the final tuple.

1. Identify Entities: Locate potential Cause and Effect.
2. Check Modality: Is it a fact or a reported claim/hedge? If hedge, discard.
3. Determine Cardinality: Does the sentence need decomposition? Implement according to the decision
4. Determine Causality Type: Check for specific keywords (variations of "due to", "results in", "led to", "effect of", "causes".) vs implicit phrasing.
5. Determine Sententiality: Are entities in the same sentence or across sentences?
6. Formulate Output: Create the tuple.

Input Text: "{sentence}"

Think step-by-step, then provide your reasoning and final output.

{ExtractionPrompts.JSON_OUTPUT_FORMAT}"""
    
    @staticmethod
    def cot_fewshot(sentence: str) -> str:
        """Hybrid (CoT + Few-Shot) prompt"""
        return f"""{ExtractionPrompts.BASE_INSTRUCTIONS}

Example:

Input: "Tumor growth was inhibited by the new compound. However, Dr. Smith claims this might cause fatigue."

Reasoning:
1. Scanning first sentence: "Tumor growth was inhibited by the new compound."
   * Entities: Cause="new compound", Effect="Tumor growth inhibition".
   * Modality Check: Factual statement. No hedging. Accepted.
   * Voice: Passive ("inhibited by"). Canonicalize to Active.
   * Causality Type: "Inhibited" is not in the explicit list ("due to", "results in", etc.). Type = Implicit.
   * Sententiality: Same sentence. Type = Intra-sentential.

2. Scanning second sentence: "However, Dr. Smith claims this might cause fatigue."
   * Modality Check: "Claims... might cause". This is a reported claim and a hedge. Rejected based on exclusion guideline.

Final Output:
{{
  "pairs": [
    {{
      "cause": "new compound",
      "effect": "Tumor growth inhibition",
      "causality": "Implicit",
      "sententiality": "Intra"
    }}
  ]
}}

Task:
Input Text: "{sentence}"

Reasoning:
[Provide your step-by-step reasoning here]

Final Output:
{ExtractionPrompts.JSON_OUTPUT_FORMAT}"""
    
    @staticmethod
    def least_to_most(sentence: str) -> str:
        """Least-to-Most prompt"""
        return f"""{ExtractionPrompts.BASE_INSTRUCTIONS}

Step 1: Entity Identification
Identify all phrases acting as potential agents (Causes) or outcomes (Effects) in the text. Ignore entities involved only in "reported claims" or "hedged" statements.

Step 2: Decomposition & Pairing
Pair the identified entities. If a compound cause or effect exists (e.g., "A and B cause C"), decide if they act independently (decompose) or jointly (do not decompose).

Step 3: Classification
For each valid pair, determine:
* Causality Type (Explicit if using "due to," "results in" "leads to," "causes", "effect of"., otherwise Implicit).
* Sententiality (Intra- vs Inter-sentential).

Step 4: Final Extraction
Generate the final list of tuples based on the findings in Steps 1-3.

Input Text: "{sentence}"

Work through each step, then provide your final output.

{ExtractionPrompts.JSON_OUTPUT_FORMAT}"""
    
    @staticmethod
    def react(sentence: str) -> str:
        """ReAct (Reason + Act) prompt"""
        return f"""{ExtractionPrompts.BASE_INSTRUCTIONS}

Instruction: Use a Thought, Action, Observation loop to process the text.

Thought 1: I need to scan the text for potential causal markers or semantic causal verbs.
Action 1: Identify candidate sentences containing causal logic.
Observation 1: [List candidate sentences]

Thought 2: I need to filter these candidates based on Modality (Fact vs. Claim/Hedge).
Action 2: Remove sentences with "may", "could", "believes", or "suggests". Keep only "can cause" or factual statements.
Observation 2: [List filtered sentences]

Thought 3: I need to apply Cardinality and Passive Voice rules to the remaining sentences.
Action 3: Decompose "AND" lists if independent. Convert passive to active.
Observation 3: [List refined pairs]

Thought 4: I need to classify Axis 1 (Explicit/Implicit) and Axis 2 (Sententiality) for the valid pairs.
Action 4: Check against the strict list ("due to", "results in", etc.) for Explicit status. Check sentence boundaries.

Input Text: "{sentence}"

Work through the Thought-Action-Observation loop, then provide your final answer.

{ExtractionPrompts.JSON_OUTPUT_FORMAT}"""
    
    @staticmethod
    def get_prompt(strategy: str, sentence: str) -> str:
        """Get prompt based on strategy"""
        strategy_map = {
            'zero-shot': ExtractionPrompts.zero_shot,
            'few-shot': ExtractionPrompts.few_shot,
            'cot': ExtractionPrompts.cot,
            'cot-fewshot': ExtractionPrompts.cot_fewshot,
            'least-to-most': ExtractionPrompts.least_to_most,
            'react': ExtractionPrompts.react
        }
        
        if strategy not in strategy_map:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(strategy_map.keys())}")
        
        return strategy_map[strategy](sentence)
    
    @staticmethod
    def parse_response(response: str) -> Dict:
        """Parse model response to extract JSON"""
        try:
            # Try to find JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx+1]
                parsed = json.loads(json_str)
                
                # Validate structure
                if 'pairs' in parsed:
                    return parsed
            
            # If no valid JSON found, return empty
            return {"pairs": []}
            
        except json.JSONDecodeError:
            return {"pairs": []}
    
    @staticmethod
    def validate_extraction(extraction: Dict) -> bool:
        """Validate extraction format"""
        if not isinstance(extraction, dict):
            return False
        if 'pairs' not in extraction:
            return False
        if not isinstance(extraction['pairs'], list):
            return False
        
        for pair in extraction['pairs']:
            required_keys = {'cause', 'effect', 'causality', 'sententiality'}
            if not all(key in pair for key in required_keys):
                return False
            
            # Validate values
            if pair['causality'] not in ['Explicit', 'Implicit']:
                return False
            if pair['sententiality'] not in ['Intra', 'Inter']:
                return False
        
        return True


def test_prompts():
    """Test all prompt templates"""
    test_sentence = "Hypertension leads to heart failure and kidney damage."
    
    strategies = ['zero-shot', 'few-shot', 'cot', 'cot-fewshot', 'least-to-most', 'react']
    
    for strategy in strategies:
        print(f"\n{'='*80}")
        print(f"Strategy: {strategy.upper()}")
        print(f"{'='*80}")
        prompt = ExtractionPrompts.get_prompt(strategy, test_sentence)
        print(prompt)


if __name__ == "__main__":
    test_prompts()