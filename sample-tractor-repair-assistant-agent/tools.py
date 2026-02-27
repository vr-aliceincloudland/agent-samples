from typing import Annotated, Sequence, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.retrievers import BaseRetriever
from langchain_tavily import TavilySearch
# from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    context: str
    relevance: str
    content: str


class AgentTools:
    """A class to hold the tools for the agent."""

    def __init__(
        self, retriever: BaseRetriever, llm: BaseChatModel, web_search_tool: TavilySearch
    ):
        self.retriever = retriever
        self.llm = llm
        self.web_search_tool = web_search_tool

    def retrieve_node(self, state: AgentState):
        """
        Retrieves relevant manual chunks based on the user's latest query.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            dict: A dictionary with the retrieved context.
        """
        print("---RETRIEVING DOCUMENTS---")
        last_message = state["messages"][-1].content
        docs = self.retriever.invoke(last_message)
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"context": context}

    def grade_documents_node(self, state: AgentState):
        """
        Determines whether the retrieved documents are relevant to the question.
        """
        print("---CHECKING DOCUMENT RELEVANCE---")

        class GradeDocuments(BaseModel):
            """Binary score for relevance check on retrieved documents."""

            binary_score: str = Field(
                description="Documents are relevant to the question, 'yes' or 'no'"
            )

        # LLM with function call
        structured_llm_grader = self.llm.with_structured_output(GradeDocuments)

        question = state["messages"][-1].content
        docs = state["context"]

        # Prompt
        system = """You are a grader assessing relevance of a retrieved document to a user question.
        If the document contains keywords related to the user question, grade it as relevant.
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        grade_prompt = [
            SystemMessage(content=system),
            HumanMessage(
                content=f"Retrieved document: \n\n {docs} \n\n User question: {question}"
            ),
        ]

        grade = structured_llm_grader.invoke(grade_prompt)

        if grade.binary_score == "yes":
            return {"relevance": "relevant"}
        else:
            return {"relevance": "not relevant"}

    def generate_node(self, state: AgentState):
        """
        Generates an answer using the retrieved context.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            dict: A dictionary with the generated message.
        """
        print("---GENERATING RESPONSE---")
        user_query = state["messages"][-1].content
        context = state.get("context", "")
        relevance = state.get("relevance", "")

        if relevance == "not relevant":
            source_info = "the web"
        else:
            source_info = "the provided manuals"

        system_prompt = (
            f"You are a Tractor Repair and Maintenance Expert. Use the following information from {source_info} "
            "to help the onsite technician fix the machine. If the answer is not in "
            "the provided information, tell them you do not know. Emphasize safety protocols.\n\n"
            f"Context:\n{context}"
        )

        response = self.llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_query)]
        )
        return {"messages": [response]}

    def web_search_node(self, state: AgentState):
        """
        Performs a web search for the user's query.
        """
        print("---PERFORMING WEB SEARCH---")
        question = state["messages"][-1].content
        docs = self.web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        return {"context": web_results}

    def decide_to_generate(self, state: AgentState):
        """
        Determines whether to generate an answer or perform a web search.
        """
        if state["relevance"] == "not relevant":
            print("---DECISION: RAG FAILED, ROUTING TO WEB SEARCH---")
            return "web_search"
        else:
            print("---DECISION: RAG SUCCEEDED, ROUTING TO GENERATE---")
            return "generate"