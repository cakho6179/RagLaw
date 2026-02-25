"""HyDE (Hypothetical Document Embeddings) utility."""

import logging

logger = logging.getLogger(__name__)


def generate_hypothetical_doc(query: str, llm) -> str:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                "Viết một đoạn văn pháp luật giả định có thể trả lời câu hỏi dưới đây:",
            ),
            ("human", "{query}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": query})
