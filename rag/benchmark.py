from niah import NIAH
from basic_rag import BasicRAG

# Get dataset
niah = NIAH()

# Get rag system
rag = BasicRAG()

for (context, question, answer) in niah:
    rag.setup(context)

    rag_result = rag.predict(question)

    eval_result = niah.evaluate(query=None, expected_answer=answer, actual_result=rag_result)

    print("rag_answer:", rag_result.answer)
    print("eval_result:", eval_result)

    rag.teardown()