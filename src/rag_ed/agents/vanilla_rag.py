"""Simple one-step retrieval agent.

The module provides a thin wrapper around :class:`~rag_ed.retrievers.vectorstore`
to perform a single retrieval and answer a query. File paths are supplied by the
caller; no paths are hard coded within the module.
"""

import argparse

import langchain_core.embeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

from rag_ed.embeddings import PassThroughEmbeddings
from rag_ed.retrievers.vectorstore import VectorStoreRetriever


def one_step_retrieval(
    query: str,
    *,
    canvas_path: str,
    piazza_path: str,
    pass_through: bool = False,
    embeddings: langchain_core.embeddings.Embeddings | None = None,
) -> str:
    """Answer ``query`` using a single retrieval step.

    Parameters
    ----------
    query:
        User question to answer.
    canvas_path:
        Path to a Canvas ``.imscc`` export.
    piazza_path:
        Path to a Piazza ``.zip`` export.
    pass_through:
        If ``True``, return retrieved documents directly instead of calling an
        LLM.
    embeddings:
        Optional embedding model. Defaults to
        :class:`langchain_openai.embeddings.OpenAIEmbeddings`.

    Returns
    -------
    str
        The answer returned by the language model or concatenated documents
        when ``pass_through`` is ``True``.
    """

    retriever = VectorStoreRetriever(
        canvas_path=canvas_path,
        piazza_path=piazza_path,
        vector_store_type="in_memory",
        embeddings=embeddings,
    )
    if pass_through:
        docs = retriever.retrieve(query)
        return "\n".join(doc.page_content for doc in docs)
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0.7, model_name="gpt-4o-mini"),
        chain_type="stuff",
        retriever=retriever.vector_store,
    )
    return qa.run(query)


def main() -> None:
    """CLI entry point for one-step retrieval."""
    parser = argparse.ArgumentParser(description="Run one-step retrieval")
    parser.add_argument("query", help="Query string")
    parser.add_argument("--canvas", required=True, help="Path to Canvas .imscc file")
    parser.add_argument("--piazza", required=True, help="Path to Piazza export .zip")
    parser.add_argument(
        "--pass-through",
        action="store_true",
        help="Return retrieved documents without calling the LLM.",
    )
    args = parser.parse_args()

    embeddings = PassThroughEmbeddings() if args.pass_through else None
    print(
        one_step_retrieval(
            args.query,
            canvas_path=args.canvas,
            piazza_path=args.piazza,
            pass_through=args.pass_through,
            embeddings=embeddings,
        )
    )


if __name__ == "__main__":
    main()
