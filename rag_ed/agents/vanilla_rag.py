from argparse import ArgumentParser
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from rag_ed.retrievers.vectorstore import VectorStoreRetriever


def one_step_retrieval(query: str) -> str:
    """Perform one-step retrieval using the specified query."""
    retriever = VectorStoreRetriever(
        canvas_path="/Users/work/Downloads/canvas.imscc",
        piazza_path="/Users/work/Downloads/piazza.zip",
        in_memory=True,
    )
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0.7, model_name="gpt-4o-mini"),
        chain_type="stuff",
        retriever=retriever.vector_store,
    )
    return qa.run(query)


def main() -> None:
    """CLI entry point for one-step retrieval."""
    parser = ArgumentParser(description="Run a vanilla RAG query.")
    parser.add_argument("query", help="Query to process.")
    args = parser.parse_args()
    print(one_step_retrieval(args.query))


if __name__ == "__main__":
    main()
