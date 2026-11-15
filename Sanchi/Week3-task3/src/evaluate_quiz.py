import json
from pathlib import Path
from difflib import SequenceMatcher

QUIZ_FILE = Path("results/generated_quiz.json")
OUTPUT_FILE = Path("results/quiz_evaluation.json")


REQUIRED_FIELDS = {
    "question",
    "answer",
    "supporting_excerpt",
    "source_file",
    "page",
    "chunk_id",
    "difficulty"
}


def similarity(a, b):
    return SequenceMatcher(
        None,
        a.lower().strip(),
        b.lower().strip()
    ).ratio()


def evaluate_questions(questions):

    results = []

    for i, q in enumerate(questions, start=1):

        checks = {}

        # 1. Required fields
        checks["complete_fields"] = (
            REQUIRED_FIELDS.issubset(q.keys())
        )

        # 2. Supporting evidence exists
        checks["evidence_present"] = bool(
            q.get("supporting_excerpt", "").strip()
        )

        # 3. Source citation exists
        checks["citation_present"] = (
            bool(q.get("source_file"))
            and q.get("page") is not None
            and bool(q.get("chunk_id"))
        )

        # 4. Answer exists
        checks["answer_present"] = bool(
            q.get("answer", "").strip()
        )

        # 5. Question exists
        checks["question_present"] = bool(
            q.get("question", "").strip()
        )

        # 6. Basic answerability
        checks["answerable"] = (
            checks["question_present"]
            and checks["answer_present"]
            and checks["evidence_present"]
        )

        # 7. Difficulty validity
        checks["valid_difficulty"] = q.get(
            "difficulty"
        ) in {"easy", "medium", "hard"}

        results.append({
            "question_number": i,
            "question": q.get("question"),
            "answer": q.get("answer"),
            "source_file": q.get("source_file"),
            "page": q.get("page"),
            "chunk_id": q.get("chunk_id"),
            "difficulty": q.get("difficulty"),
            "checks": checks
        })

    # Duplicate-question detection
    for i in range(len(results)):

        duplicate = False

        for j in range(i):

            score = similarity(
                results[i]["question"],
                results[j]["question"]
            )

            if score >= 0.85:
                duplicate = True
                break

        results[i]["checks"]["duplicate_question"] = duplicate

    return results


def main():

    if not QUIZ_FILE.exists():
        raise FileNotFoundError(
            "generated_quiz.json not found."
        )

    with open(
        QUIZ_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        quiz = json.load(f)

    questions = quiz.get("questions", [])

    print(
        f"\nEvaluating {len(questions)} questions..."
    )

    results = evaluate_questions(questions)

    summary = {
        "total_questions": len(results),
        "complete_fields": sum(
            r["checks"]["complete_fields"]
            for r in results
        ),
        "evidence_present": sum(
            r["checks"]["evidence_present"]
            for r in results
        ),
        "citation_present": sum(
            r["checks"]["citation_present"]
            for r in results
        ),
        "answerable": sum(
            r["checks"]["answerable"]
            for r in results
        ),
        "valid_difficulty": sum(
            r["checks"]["valid_difficulty"]
            for r in results
        ),
        "duplicates": sum(
            r["checks"]["duplicate_question"]
            for r in results
        )
    }

    output = {
        "summary": summary,
        "questions": results
    }

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
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\nQUALITY SUMMARY")
    print("-" * 50)

    print(
        f"Total questions: "
        f"{summary['total_questions']}"
    )

    print(
        f"Complete fields: "
        f"{summary['complete_fields']}/"
        f"{summary['total_questions']}"
    )

    print(
        f"Evidence present: "
        f"{summary['evidence_present']}/"
        f"{summary['total_questions']}"
    )

    print(
        f"Citation present: "
        f"{summary['citation_present']}/"
        f"{summary['total_questions']}"
    )

    print(
        f"Answerable: "
        f"{summary['answerable']}/"
        f"{summary['total_questions']}"
    )

    print(
        f"Valid difficulty: "
        f"{summary['valid_difficulty']}/"
        f"{summary['total_questions']}"
    )

    print(
        f"Potential duplicates: "
        f"{summary['duplicates']}"
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
