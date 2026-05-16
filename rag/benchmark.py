from niah import NIAH
from basic_rag import BasicRAG

# Get dataset
niah = NIAH()

# Get rag system
rag = BasicRAG()

for (context, question, answer) in niah:
    print("question:", question)
    
    rag.setup(context)

    rag_result = rag.predict(question)

    eval_result = niah.evaluate(query=None, expected_answer=answer, actual_result=rag_result)

    for i, (text, score) in enumerate(rag_result.retrieved_context):
        print(f"Retrieved Chunk {i+1} - score {score}: {text}")
        print()

    print("rag_answer:", rag_result.answer)
    print("eval_result:", eval_result)

    print("-"*10)

    rag.teardown()