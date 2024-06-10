import snowflake.connector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


import config
from llm import Cortex
from ner import EntityFinder
from graph import NewsGraphClient

snowflake_connection = snowflake.connector.connect(**config.SNOWFLAKE_CONNECTION_PARAMS)
model = Cortex(connection=snowflake_connection, model=config.CHAT_MODEL)
entity_finder = EntityFinder(config.RELEVANT_LABELS)
db = NewsGraphClient()


CYPHER_GENERATION_TEMPLATE = """Based on the graph schema below, write a Cypher query that answers the user's question. 
Use only the node labels, relationships and properties provided in the schema:
{schema}
Entities in the question map to the following database values:
{entities_list}

Here are some examples: 
Example 1: For the question "List 10 titles of articles mentioning Ursula von der Leyen" and the entity list "(:Person {{ name: 'Ursula von der Leyen' }}, (:Person {{ name: 'Ursula v. d. Leyn' }}" the generated Cypher query should be 
"MATCH (a:Article)-[:CONTAINS]->(c:Chunk)-[:MENTIONS]->(o:Person) WHERE o.name IN ['Ursula von der Leyen', 'Ursula v. d. Leyn'] RETURN DISTINCT a.title LIMIT 10"

Example 2: For the question "How many sources mention the EU commission?" and the entity list "(:Organization {{ name: 'EU-Kommission' }}" the generated Cypher query should be 
"MATCH (s:Source)-[:PUBLISHED]->(a:Article)-[:CONTAINS]->(c:Chunk)-[:MENTIONS]->(o:Organization) WHERE o.name IN ['EU-Kommission'] WITH DISTINCT s RETURN count(s)"

Example 3: For the question "News about France and Macron?" and the entity list "(:Location {{ name: 'France' }}, (:Person {{ name: 'Emmanuel Macron' }}" the generated Cypher query should be 
"MATCH (c:Chunk)-[:MENTIONS]->(o:Location) WHERE o.name = 'France' UNION MATCH (c:Chunk)-[:MENTIONS]->(o:Person) WHERE o.name = 'Emmanuel Macron' RETURN c.text LIMIT 10"

Question: {question}
Cypher query:"""

ANSWER_PROMPT_TEMPLATE = (
    "Answer the question below in appropriate detail, given the following context. "
    # "Think step by step before providing a detailed answer. "
    "The context was retrieved from the database by the following query:\n\n"
    "Query: {query}\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer: "
)


def generate_cypher_query(question: str) -> str:
    # Get entities from text
    mentioned_entities = entity_finder.find(question)
    # Perform fulltext search
    candidates = db.lookup_mentioned_entities(mentioned_entities)
    candidate_context = map_candidates_to_context(candidates)
    # Define prompt
    cypher_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given an input question, convert it to a Cypher query. No pre-amble.",),
        ("human", CYPHER_GENERATION_TEMPLATE),
    ])
    # Define chain
    cypher_chain = cypher_prompt | model | StrOutputParser()
    # Generate Cypher query with found entities
    generated_query = cypher_chain.invoke({
        'question': question,
        'entities_list': candidate_context,
        'schema': db.graph.schema
    })
    return generated_query


def answer_question(question: str, generated_query: str):
    # Perform query
    response = db.query(generated_query)
    context = map_records_to_context(response)
    # Define prompt and chain
    answer_prompt = ChatPromptTemplate.from_template(ANSWER_PROMPT_TEMPLATE)
    answer_chain = answer_prompt | model | StrOutputParser()
    # Populate context and generate answer
    answer = answer_chain.invoke(
        {'question': question, 'context': context, 'query': generated_query}
    )
    return answer


# Helper functions to map retrieved values to context strings

def map_candidates_to_context(candidates: list[dict[str, str]]) -> str:
    context_str = ', '.join(
        f"(:{c['label']} {{ name: '{c['name']}' }}"
        for c in candidates
    )
    return context_str


def map_records_to_context(db_records: list[dict]) -> str:
    context_str = (f"\n{'='*5}\n").join(
        '/n'.join(f"{k}: {v}" for k, v in record.items())
        for record in db_records
    )
    return context_str


if __name__ == "__main__":
    # Steps:
    # Get input query
    # question = input('Pose a question to the LLM: ')
    # Possible questions:
    question = 'When was the article with the title "Hochrechnung zur Europawahl 2024: AfD in Ostdeutschland stärkste Kraft" published?'
    question = 'List 5 article titles about Volt'
    question = 'How many sources mention the EU parliament?'
    question = 'What do the news have to say about Olaf Scholz?'
    # Generate query
    generated_query = generate_cypher_query(question)
    print(generated_query)
    # Generate answer
    answer = answer_question(question, generated_query)
    print(answer)