"""
Prompt Shot Classifier & Heuristic Detector
Analyzes prompts to classify them as Zero-shot, One-shot, or Few-shot.
Provides heuristic pattern matching, example extraction, and structural diagnostics.
"""

import re
from typing import Dict, Any, List, Tuple


class PromptShotDetector:
    """
    Heuristic and structural analyzer to determine whether a prompt is
    Zero-shot, One-shot, or Few-shot.
    """

    # Common pattern pairs indicating demonstration input/output
    PAIR_PATTERNS = [
        # Explicit Example tags: Example 1:, Example 2:, etc.
        (r'(?:Example|Ex|Sample)\s*#?\d*[:\.-]?\s*(.*?)(?=(?:Example|Ex|Sample)\s*#?\d*[:\.-]?|\Z)', 'example_blocks'),
        # XML-style tags <example>...</example>
        (r'<example>\s*(.*?)\s*</example>', 'xml_examples'),
        # Input / Output pairs
        (r'(?:Input|Query|Text|Sentence|Premise|Context|Tweet|Code|Review)\s*[:\.-]\s*(.*?)\s*(?:Output|Response|Label|Answer|Result|Sentiment|Translation|Target|Summary)\s*[:\.-]\s*([^\n]+(?:\n(?!(?:Input|Query|Text|Sentence|Premise|Context|Tweet|Code|Review)\s*[:\.-]).*)*)', 'io_pairs'),
        # Q: and A: pairs
        (r'(?:Question|Q)\s*[:\.-]\s*(.*?)\s*(?:Answer|A)\s*[:\.-]\s*([^\n]+(?:\n(?!(?:Question|Q)\s*[:\.-]).*)*)', 'qa_pairs'),
        # Arrow notation: "A -> B" or "A => B"
        (r'([^\n\->=]+)\s*(?:->|=>|→|↦)\s*([^\n]+)', 'arrow_pairs'),
    ]

    # Target completion indicators at the end of prompt
    TARGET_PROMPT_PATTERNS = [
        r'(?:Input|Query|Text|Sentence|Tweet|Review|Question|Q)\s*[:\.-]\s*(.*?)\s*(?:Output|Response|Label|Answer|Result|Sentiment|Translation|A)\s*[:\.-]\s*$',
        r'(?:Output|Response|Answer|Result|A)\s*[:\.-]\s*$',
        r'(?:->|=>|→|↦)\s*$',
    ]

    @classmethod
    def analyze(cls, prompt: str) -> Dict[str, Any]:
        """
        Analyze prompt text and return classification, extracted examples, and diagnostics.
        """
        prompt = prompt.strip()
        if not prompt:
            return {
                "classification": "zero-shot",
                "shot_count": 0,
                "confidence": 1.0,
                "reasoning": "Prompt is empty.",
                "examples": [],
                "target_query": "",
                "patterns_found": [],
                "prompt_metrics": {
                    "char_count": 0,
                    "word_count": 0,
                    "line_count": 0
                }
            }

        lines = [line.strip() for line in prompt.split('\n') if line.strip()]
        char_count = len(prompt)
        word_count = len(prompt.split())
        line_count = len(lines)

        detected_examples: List[Dict[str, str]] = []
        patterns_matched = []

        # 1. Check for XML example tags <example>...</example>
        xml_matches = re.findall(r'<example>(.*?)</example>', prompt, flags=re.DOTALL | re.IGNORECASE)
        if xml_matches:
            patterns_matched.append("XML <example> tags")
            for idx, ex in enumerate(xml_matches):
                detected_examples.append({
                    "id": idx + 1,
                    "type": "xml_block",
                    "content": ex.strip(),
                    "input": cls._extract_subpart(ex, ['input', 'q', 'question', 'text']),
                    "output": cls._extract_subpart(ex, ['output', 'a', 'answer', 'label', 'result'])
                })

        # 2. Check for explicit Example 1:, Example 2:, etc.
        if not detected_examples:
            num_ex_matches = list(re.finditer(r'(?:Example|Ex|Sample)\s*#?(\d+)[:\.-]\s*(.*?)(?=(?:Example|Ex|Sample)\s*#?\d*[:\.-]|(?:Now|Task|Test|Actual|Query|Input)[:\.-]|\Z)', prompt, flags=re.DOTALL | re.IGNORECASE))
            if num_ex_matches:
                patterns_matched.append("Numbered Example blocks (e.g. Example 1, Example 2)")
                for idx, match in enumerate(num_ex_matches):
                    content = match.group(2).strip()
                    detected_examples.append({
                        "id": idx + 1,
                        "type": "numbered_example",
                        "content": content,
                        "input": cls._extract_subpart(content, ['input', 'q', 'question', 'text', 'premise']),
                        "output": cls._extract_subpart(content, ['output', 'a', 'answer', 'label', 'result', 'sentiment'])
                    })

        # 3. Check for Input: / Output: or Q: / A: pairs
        if not detected_examples:
            # Match I/O or Q/A pairs, stopping before the next prompt marker (whether on a new line or inline)
            io_matches = list(re.finditer(
                r'(?P<in_label>Input|Query|Text|Sentence|Premise|Context|Tweet|Code|Review|Question|Q)\s*[:\.-]\s*(?P<in_val>.*?)\s*(?P<out_label>Output|Response|Label|Answer|Result|Sentiment|Translation|Target|Summary|A)\s*[:\.-]\s*(?P<out_val>.*?)(?=(?:Input|Query|Text|Sentence|Premise|Context|Tweet|Code|Review|Question|Q)\s*[:\.-]|\Z)',
                prompt,
                flags=re.IGNORECASE | re.DOTALL
            ))
            
            # Check if the last occurrence is an unresolved query (no out_val or at the very end with blank answer)
            if io_matches:
                patterns_matched.append("Input/Output or Q/A pair labels")
                for idx, match in enumerate(io_matches):
                    in_text = match.group('in_val').strip()
                    out_text = match.group('out_val').strip()
                    
                    # If this is the last item and output is empty or asking model to complete, it's the target query, not an example
                    is_last = (idx == len(io_matches) - 1)
                    if is_last and (len(out_text) == 0 or out_text.lower() in ['[insert answer]', '?', '...', '[blank]', '']):
                        continue
                    
                    if out_text:
                        detected_examples.append({
                            "id": len(detected_examples) + 1,
                            "type": "io_pair",
                            "input": in_text,
                            "output": out_text,
                            "content": f"{match.group('in_label')}: {in_text}\n{match.group('out_label')}: {out_text}"
                        })

        # 4. Check for arrow pairs "text -> label" or "A => B"
        if not detected_examples:
            arrow_matches = list(re.finditer(r'^([^\n\->=]+?)\s*(?:->|=>|→)\s*([^\n]+)$', prompt, flags=re.MULTILINE))
            if len(arrow_matches) >= 1:
                # Filter out lines that look like final unsolved questions (ending in empty or '?')
                valid_arrows = []
                for m in arrow_matches:
                    left = m.group(1).strip()
                    right = m.group(2).strip()
                    if right and right not in ['?', '...']:
                        valid_arrows.append((left, right))
                
                if valid_arrows:
                    patterns_matched.append("Arrow demonstration format (A -> B)")
                    for idx, (left, right) in enumerate(valid_arrows):
                        detected_examples.append({
                            "id": idx + 1,
                            "type": "arrow_mapping",
                            "input": left,
                            "output": right,
                            "content": f"{left} -> {right}"
                        })

        # 5. Extract target query (the actual task given to model to solve)
        target_query = cls._detect_target_query(prompt, detected_examples)

        # Count detected examples
        example_count = len(detected_examples)

        # Classify based on count
        if example_count == 0:
            classification = "zero-shot"
            confidence = 0.95 if not patterns_matched else 0.80
            reasoning = (
                "No demonstration input-output examples were detected in the prompt. "
                "The prompt asks the model to perform the task directly based solely on instructions "
                "or context without prior solved instances."
            )
        elif example_count == 1:
            classification = "one-shot"
            confidence = 0.90
            reasoning = (
                "Exactly 1 demonstration example was detected. The prompt provides a single solved "
                "input-output sample to guide the model's response formatting or reasoning before the actual task."
            )
        else:
            classification = "few-shot"
            confidence = 0.95
            reasoning = (
                f"{example_count} demonstration examples were detected. The prompt uses multiple solved "
                f"instances (few-shot prompting) to establish the expected output format, reasoning, and pattern."
            )

        return {
            "classification": classification,
            "shot_count": example_count,
            "confidence": confidence,
            "reasoning": reasoning,
            "examples": detected_examples,
            "target_query": target_query,
            "patterns_found": patterns_matched,
            "prompt_metrics": {
                "char_count": char_count,
                "word_count": word_count,
                "line_count": line_count,
            }
        }

    @staticmethod
    def _extract_subpart(text: str, keys: List[str]) -> str:
        """Helper to extract input or output value from a block of text."""
        pattern = r'(?:' + '|'.join(keys) + r')\s*[:\.-]\s*(.*?)(?=(?:' + '|'.join(keys) + r')\s*[:\.-]|\Z)'
        m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
        return text.strip()

    @staticmethod
    def _detect_target_query(prompt: str, examples: List[Dict[str, Any]]) -> str:
        """Find the final task or query given to the LLM to complete."""
        # Check standard trailing patterns
        lines = [l for l in prompt.split('\n') if l.strip()]
        if not lines:
            return ""

        # If examples were extracted, the target query is likely in the text after the last example
        if examples:
            last_ex = examples[-1].get("content", "")
            if last_ex:
                idx = prompt.rfind(last_ex)
                if idx != -1:
                    remainder = prompt[idx + len(last_ex):].strip()
                    if remainder:
                        return remainder

        # Otherwise check the last 1-3 lines
        return "\n".join(lines[-2:]) if len(lines) >= 2 else lines[-1]
