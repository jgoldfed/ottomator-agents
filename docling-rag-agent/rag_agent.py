"""
RAG CLI Agent with PostgreSQL/PGVector
=======================================
Text-based CLI agent that searches through knowledge base using semantic similarity
"""

import asyncio
import asyncpg
import json
import logging
import os
import sys
from typing import Any

from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext

from utils.providers import get_llm_model

# Load environment variables
load_dotenv(".env")

logger = logging.getLogger(__name__)

# Global database pool
db_pool = None


async def initialize_db():
    """Initialize database connection pool."""
    global db_pool
    if not db_pool:
        db_pool = await asyncpg.create_pool(
            os.getenv("DATABASE_URL"),
            min_size=2,
            max_size=10,
            command_timeout=60
        )
        logger.info("Database connection pool initialized")


async def close_db():
    """Close database connection pool."""
    global db_pool
    if db_pool:
        await db_pool.close()
        logger.info("Database connection pool closed")


async def search_knowledge_base(ctx: RunContext[None], query: str, limit: int = 5) -> str:
    """
    Search the knowledge base using semantic similarity.

    Args:
        query: The search query to find relevant information
        limit: Maximum number of results to return (default: 5)

    Returns:
        Formatted search results with source citations
    """
    try:
        # Ensure database is initialized
        if not db_pool:
            await initialize_db()

        # Generate embedding for query
        from ingestion.embedder import create_embedder
        embedder = create_embedder()
        query_embedding = await embedder.embed_query(query)

        # Convert to PostgreSQL vector format
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

        # Search using match_chunks function
        async with db_pool.acquire() as conn:
            results = await conn.fetch(
                """
                SELECT * FROM match_chunks($1::vector, $2)
                """,
                embedding_str,
                limit
            )

        # Format results for response
        if not results:
            return "No relevant information found in the knowledge base for your query."

        # Build response with sources
        response_parts = []
        for i, row in enumerate(results, 1):
            similarity = row['similarity']
            content = row['content']
            doc_title = row['document_title']
            doc_source = row['document_source']

            response_parts.append(
                f"[Source: {doc_title}]\n{content}\n"
            )

        if not response_parts:
            return "Found some results but they may not be directly relevant to your query. Please try rephrasing your question."

        return f"Found {len(response_parts)} relevant results:\n\n" + "\n---\n".join(response_parts)
    except Exception as e:
        logger.error(f"Knowledge base search failed: {e}", exc_info=True)
        return f"I encountered an error searching the knowledge base: {str(e)}"


# Create the PydanticAI agent with the RAG tool
agent = Agent(
    get_llm_model(),
    system_prompt="""# CONTEXT
You are an expert researcher and proposal writer for AVER LLC, with deep expertise in government proposal development. You have access to AVER's comprehensive documentation repository, which includes:
- Past successful proposals
- Proposal templates and frameworks 
- Best practices guides
- Employee information and capabilities
- Government contracting requirements and standards

# GOAL
Your primary objective is to serve as a proposal development assistant who:
1. Provides accurate, actionable guidance based on AVER's documented best practices
2. Generates proposal-ready content that can be directly used in future submissions
3. Ensures all responses align with AVER's standards and government contracting requirements

# RESPONSE FORMAT
Provide clear, detailed responses that follow proposal writing best practices and government contracting standards.

# CRITICAL INSTRUCTIONS
1. Be proactive - take necessary actions without asking for user permission
2. Always check documentation first:
   - Begin with RAG search
   - Review full documentation page list
   - Retrieve and analyze relevant content

3. **IMPORTANT**: After using the search_knowledge_base tool, you MUST provide a complete answer based on the results.
   - Never just call tools without providing a final answer to the user
   - Synthesize the information from search results into a clear, comprehensive response
   - Always end with a clear, direct answer to the user's question

4. Content Requirements:
   - Be detailed and specific in all responses
   - Ensure answers can be directly used in proposals
   - Maintain consistency with AVER's standards
   - Follow government contracting requirements

5. Communication Style:
   - Be clear and professional
   - Focus on practical, actionable guidance
   - Maintain proposal-appropriate language and tone

6. Document Processing:
   - Thoroughly analyze all documents provided in the context
   - Extract relevant information from PDFs and documents including text, tables, charts, and graphics
   - Reference specific sources when using information from them (e.g., "According to the XYZ Report, page 12...")
   - Prioritize recent documents when information conflicts
   - Identify and flag any inconsistencies between content and other knowledge base sources
   - When appropriate, suggest how visual elements from the documents could be adapted for proposals""",
    tools=[search_knowledge_base],
    retries=2
)


async def run_cli():
    """Run the agent in an interactive CLI with streaming."""

    # Initialize database
    await initialize_db()

    print("=" * 60)
    print("RAG Knowledge Assistant")
    print("=" * 60)
    print("Ask me anything about the knowledge base!")
    print("Type 'quit', 'exit', or press Ctrl+C to exit.")
    print("=" * 60)
    print()

    message_history = []

    try:
        while True:
            # Get user input
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nAssistant: Thank you for using the knowledge assistant. Goodbye!")
                break

            print("Assistant: ", end="", flush=True)

            try:
                # Stream the response using run_stream
                async with agent.run_stream(
                    user_input,
                    message_history=message_history
                ) as result:
                    streamed_anything = False
                    streamed_buffer: list[str] = []
                    # Stream text as it comes in (delta=True for only new tokens)
                    async for text in result.stream_text(delta=True):
                        streamed_anything = True
                        streamed_buffer.append(text)
                        # Print only the new token
                        print(text, end="", flush=True)

                    streamed_text = ''.join(streamed_buffer)
                    if (not streamed_anything) or not streamed_text.strip():
                        final_text = await result.get_output()
                        final_text_str = final_text.strip() if isinstance(final_text, str) else str(final_text)
                        if final_text_str:
                            print(final_text_str, end="", flush=True)

                    print()  # New line after streaming completes

                    # Update message history for context
                    message_history = result.all_messages()

            except KeyboardInterrupt:
                print("\n\n[Interrupted]")
                break
            except Exception as e:
                print(f"\n\nError: {e}")
                logger.error(f"Agent error: {e}", exc_info=True)

            print()  # Extra line for readability

    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    finally:
        await close_db()


async def main():
    """Main entry point."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Check required environment variables
    if not os.getenv("DATABASE_URL"):
        logger.error("DATABASE_URL environment variable is required")
        sys.exit(1)

    # Run the CLI
    await run_cli()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nShutting down...")