import argparse

from niah import NIAH
from basic_rag import BasicRAG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OnePassRAG benchmark on a selected dataset.")
    parser.add_argument(
        "--dataset",
        default="250K",
        help="Dataset subdirectory under datasets/Prism-Data (e.g. 128K, 250K)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Get dataset
    niah = NIAH(dataset_name=args.dataset)

    # Get rag system
    rag = BasicRAG()

    print(f"Running dataset: {args.dataset}")
    print(f"Dataset path: {niah.directory_path}")
    print(f"Total files: {len(niah)}")

    for (file_path, context, question, answer) in niah:
        print("file_path:", file_path)
        print("question:", question)

        rag.setup(context)

        rag_result = rag.predict(question)

        eval_result = niah.evaluate(query=None, expected_answer=answer, actual_result=rag_result)

        for i, (text, score) in enumerate(rag_result.retrieved_context):
            print(f"Retrieved Chunk {i+1} - score {score}: {text}")
            print()

        print("rag_answer:", rag_result.answer)
        print("eval_result:", eval_result)

        print("-" * 10)

        rag.teardown()


if __name__ == "__main__":
    main()
