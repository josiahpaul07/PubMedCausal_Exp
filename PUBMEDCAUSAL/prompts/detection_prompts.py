"""
Prompt Templates for Causal Detection Task
Strategies: zero-shot, cot, few-shot, cot-fewshot, react, least-to-most
"""

from typing import Dict, List


class DetectionPrompts:
    """Prompt templates for causal detection"""
    
    # Few-shot examples
    EXAMPLES = [
        {
            "sentence": "Marketing exposure is one causal factor for adolescent smoking and e-cigarette use.",
            "reasoning": "This sentence explicitly states that marketing exposure is a 'causal factor' for smoking and e-cigarette use. The word 'causal factor' directly indicates a cause-effect relationship.",
            "label": "Yes"
        },
        {
            "sentence": "The study examined the relationship between internet usage and BMI among the elderly.",
            "reasoning": "While this mentions a 'relationship', it does not explicitly state or imply causation. It could be correlational or just an examination of association.",
            "label": "No"
        },
        {
            "sentence": "Image characteristics significantly improve fundraising results.",
            "reasoning": "The word 'improve' suggests that image characteristics lead to better fundraising results, indicating a causal relationship.",
            "label": "Yes"
        }
    ]
    
    @staticmethod
    def zero_shot(sentence: str) -> str:
        """Zero-shot prompt"""
        return f"""Determine whether the following sentence contains a causal relationship (cause-effect).

Sentence: "{sentence}"

Answer with only "Yes" or "No"."""
    
    @staticmethod
    def cot(sentence: str) -> str:
        """Chain-of-Thought prompt"""
        return f"""Determine whether the following sentence contains a causal relationship (cause-effect).

Sentence: "{sentence}"

Let's think step by step:
1. Identify if there are any cause-effect indicators (words like: cause, lead to, result in, affect, influence, due to, because, etc.)
2. Check if one event/factor is described as producing or influencing another
3. Determine if causality is present

Based on your reasoning, does this sentence contain causality?
Answer with only "Yes" or "No"."""
    
    @staticmethod
    def few_shot(sentence: str) -> str:
        """Few-shot prompt with examples"""
        examples_text = ""
        for i, ex in enumerate(DetectionPrompts.EXAMPLES, 1):
            examples_text += f"""Example {i}:
Sentence: "{ex['sentence']}"
Answer: {ex['label']}

"""
        
        return f"""{examples_text}Now analyze this sentence:

Sentence: "{sentence}"

Answer with only "Yes" or "No"."""
    
    @staticmethod
    def cot_fewshot(sentence: str) -> str:
        """Chain-of-Thought + Few-shot prompt"""
        examples_text = ""
        for i, ex in enumerate(DetectionPrompts.EXAMPLES, 1):
            examples_text += f"""Example {i}:
Sentence: "{ex['sentence']}"
Reasoning: {ex['reasoning']}
Answer: {ex['label']}

"""
        
        return f"""{examples_text}Now analyze this sentence:

Sentence: "{sentence}"

Think step by step and then answer with only "Yes" or "No"."""
    
    @staticmethod
    def react(sentence: str) -> str:
        """ReAct (Reasoning + Acting) prompt"""
        return f"""Task: Determine if the sentence contains a causal relationship.

Sentence: "{sentence}"

Thought 1: I need to identify causal indicators or relationships in this sentence.
Action 1: Scan for causal keywords (cause, effect, lead to, result in, influence, etc.)
Observation 1: [Identify what you find]

Thought 2: I should check if one factor is described as producing another.
Action 2: Analyze the sentence structure for cause-effect patterns
Observation 2: [Identify relationships]

Thought 3: Now I can make a final determination.
Action 3: Conclude whether causality is present
Answer: [Yes/No]

Provide your reasoning and final answer (Yes or No)."""
    
    @staticmethod
    def least_to_most(sentence: str) -> str:
        """Least-to-Most decomposition prompt"""
        return f"""You are an expert causal relation detection system. Your task is to find instances of causality in text. If there is even one such instance, you output "Yes", if there is none, "No".

Objective: Detect causal instances in the provided text and output the result.

Definition of Causality:
- Causation is the "highest form of association"
- There must be a strict difference between the Cause (agent/trigger) and Effect (outcome)
- "Can cause" is accepted as it shows a causal relationship
- "May cause, could cause" and other hedging are NOT causal facts
- Relations based on "reported claims" (e.g., "Scientists believe X causes Y") should be ignored
- Annotate based on semantic structure with only common sense interpretation

Input Text: "{sentence}"

Step 1: Entity Identification
Identify all phrases acting as potential agents (Causes) or outcomes (Effects) in the text.

Step 2: Causal Relation Identification
- Ignore entities involved only in "reported claims" or "hedged" statements
- Accept relations where one entity directly causes or influences another
- Look for causal indicators: cause, lead to, result in, affect, influence, produce, etc.

Step 3: Final Verdict
Based on Steps 1 and 2, determine if there is at least one instance of a causal relation.

Output: Answer with only "Yes" or "No"."""
    
    @staticmethod
    def get_prompt(strategy: str, sentence: str) -> str:
        """Get prompt based on strategy"""
        strategy_map = {
            'zero-shot': DetectionPrompts.zero_shot,
            'cot': DetectionPrompts.cot,
            'few-shot': DetectionPrompts.few_shot,
            'cot-fewshot': DetectionPrompts.cot_fewshot,
            'react': DetectionPrompts.react,
            'least-to-most': DetectionPrompts.least_to_most
        }
        
        if strategy not in strategy_map:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(strategy_map.keys())}")
        
        return strategy_map[strategy](sentence)
    
    @staticmethod
    def parse_response(response: str) -> str:
        """Parse model response to extract Yes/No answer"""
        response = response.strip().lower()
        
        # Direct yes/no
        if response in ['yes', 'no']:
            return response.capitalize()
        
        # Extract from longer responses
        if 'yes' in response and 'no' not in response:
            return 'Yes'
        elif 'no' in response and 'yes' not in response:
            return 'No'
        
        # Look for "Answer: Yes" or "Answer: No"
        if 'answer:' in response:
            answer_part = response.split('answer:')[-1].strip()
            if 'yes' in answer_part[:10]:
                return 'Yes'
            if 'no' in answer_part[:10]:
                return 'No'
        
        # Default to looking at the last word
        words = response.split()
        if words:
            last_word = words[-1].strip('.,!?')
            if last_word in ['yes', 'no']:
                return last_word.capitalize()
        
        return 'Unknown'


def test_prompts():
    """Test all prompt templates"""
    test_sentence = "Air pollution causes respiratory diseases."
    
    strategies = ['zero-shot', 'cot', 'few-shot', 'cot-fewshot', 'react', 'least-to-most']
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy.upper()}")
        print(f"{'='*60}")
        prompt = DetectionPrompts.get_prompt(strategy, test_sentence)
        print(prompt)


if __name__ == "__main__":
    test_prompts()