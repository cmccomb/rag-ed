from argparse import ArgumentParser
from smolagents import Tool, OpenAIServerModel, CodeAgent
from rag_ed.retrievers.vectorstore import VectorStoreRetriever


class RetrieverTool(Tool):
    name = "retriever"
    description = (
        "Uses semantic search to retrieve the parts of transformers documentation that could be most relevant "
        "to answer your query."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": (
                "The query to perform. This should be semantically close to your target documents. "
                "Use the affirmative form rather than a question."
            ),
        }
    }
    output_type = "string"

    def __init__(self, canvas_path, piazza_path, **kwargs):
        super().__init__(**kwargs)
        self.retriever = VectorStoreRetriever(canvas_path, piazza_path, in_memory=True)

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.retriever.retrieve(query, 5)
        return "\nRetrieved documents:\n" + "".join(
            [
                f"\n\n===== Document {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )


retriever_tool = RetrieverTool(
    canvas_path="/Users/work/Downloads/canvas.imscc",
    piazza_path="/Users/work/Downloads/piazza.zip",
)

agent = CodeAgent(
    tools=[retriever_tool], model=OpenAIServerModel(), max_steps=4, verbosity_level=2
)


def main() -> None:
    """CLI entry point for the self-querying retriever agent."""
    parser = ArgumentParser(description="Self-querying retriever agent demo.")
    parser.add_argument("query", help="Query to perform.")
    args = parser.parse_args()
    print(agent.run(args.query))


if __name__ == "__main__":
    main()
