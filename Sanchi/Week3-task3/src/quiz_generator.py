import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai


load_dotenv(override=True)

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=API_KEY)

OUTPUT_FILE = Path("results/generated_quiz.json")


def build_prompt(topic, question_count, evidence):

    evidence_text = ""

    for i, item in enumerate(evidence, start=1):

        evidence_text += f"""
EVIDENCE {i}

Document: {item["source_file"]}
Page: {item["page"]}
Chunk ID: {item["chunk_id"]}

Text:
{item["text"]}

--------------------------------
"""

    return f"""
You are a grounded educational quiz generator.

Topic / Learning Objective:
{topic}

Required number of questions:
{question_count}

Use ONLY the evidence provided below.
Do NOT use outside knowledge.

Generate exactly {question_count} short-answer questions.

Every question must contain:
- question
- answer
- supporting_excerpt
- source_file
- page
- chunk_id
- difficulty

Rules:
1. Every answer must be directly supported by the evidence.
2. Do not invent facts.
3. Do not use outside knowledge.
4. Do not create duplicate questions.
5. Keep every question relevant to the topic.
6. Keep questions clear and answerable.
7. supporting_excerpt must actually support the answer.
8. source_file, page, and chunk_id must match the evidence.
9. If the evidence does not support enough valid questions, generate only
   supported questions and set "insufficient_evidence" to true.
10. Try to maintain balanced coverage across the requested topics when the
    evidence contains multiple topics.
11. Return ONLY a JSON object. Do not use Markdown code fences.

Required JSON format:

{{
  "questions": [
    {{
      "question": "...",
      "answer": "...",
      "supporting_excerpt": "...",
      "source_file": "...",
      "page": 1,
      "chunk_id": "...",
      "difficulty": "easy"
    }}
  ],
  "insufficient_evidence": false
}}

Evidence:

{evidence_text}
"""


def clean_json_response(text):

    text = text.strip()

    if text.startswith("```json"):
        text = text[len("```json"):]

    elif text.startswith("```"):
        text = text[len("```"):]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


def generate_quiz(topic, question_count, evidence):

    prompt = build_prompt(
        topic,
        question_count,
        evidence
    )

    print(f"\nPrompt size: {len(prompt)} characters")
    print(f"Evidence chunks: {len(evidence)}")

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json"
        }
    )

    text = response.text.strip()

    text = clean_json_response(text)

    try:
        quiz = json.loads(text)

    except json.JSONDecodeError as e:

        print("\nGemini returned invalid JSON:")
        print(text)

        raise ValueError(
            f"Could not parse Gemini response as JSON: {e}"
        )

    return quiz


def validate_quiz(quiz):

    if not isinstance(quiz, dict):
        raise ValueError(
            "Quiz output must be a JSON object."
        )

    if "questions" not in quiz:
        raise ValueError(
            "Quiz output missing 'questions'."
        )

    if not isinstance(quiz["questions"], list):
        raise ValueError(
            "'questions' must be a list."
        )

    required_fields = {
        "question",
        "answer",
        "supporting_excerpt",
        "source_file",
        "page",
        "chunk_id",
        "difficulty"
    }

    for i, question in enumerate(
        quiz["questions"],
        start=1
    ):

        missing = required_fields - set(
            question.keys()
        )

        if missing:
            raise ValueError(
                f"Question {i} is missing fields: {missing}"
            )

    return True


def main():

    topic = input(
        "\nEnter topic / learning objective: "
    ).strip()

    question_count = int(
        input("Number of questions: ")
    )

    evidence_file = Path(
        "results/selected_context.json"
    )

    if not evidence_file.exists():

        raise FileNotFoundError(
            "selected_context.json not found."
        )

    with open(
        evidence_file,
        "r",
        encoding="utf-8"
    ) as f:

        evidence = json.load(f)

    # topic_context_selection.py stores the actual
    # evidence chunks inside "selected_chunks".
    if isinstance(evidence, dict):

        if "selected_chunks" not in evidence:

            raise ValueError(
                "selected_context.json does not contain "
                "'selected_chunks'."
            )

        evidence = evidence["selected_chunks"]

    if not evidence:

        raise ValueError(
            "No evidence found in selected_context.json."
        )

    print("\nGenerating grounded quiz...")

    quiz = generate_quiz(
        topic,
        question_count,
        evidence
    )

    validate_quiz(quiz)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            quiz,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\nQuiz generated successfully.")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nGenerated questions:")

    for i, question in enumerate(
        quiz["questions"],
        start=1
    ):

        print(
            f"\n{i}. {question['question']}"
        )

        print(
            f"   Answer: {question['answer']}"
        )

        print(
            f"   Source: "
            f"{question['source_file']} "
            f"(page {question['page']}, "
            f"{question['chunk_id']})"
        )


if __name__ == "__main__":
    main()