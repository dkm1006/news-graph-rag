from haystack import Pipeline, component
from haystack.utils import Secret

from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator

from news_graph_rag import config
from news_graph_rag.ner import EntityFinder
from news_graph_rag.graph import NewsGraphClient
from news_graph_rag.schema import Entity


CYPHER_GENERATION_TEMPLATE = """Based on the graph schema below, write a Cypher query that answers the user's question. 
Answer with the query and nothing else.
Use only the node labels, relationships and properties provided in the schema:
{{schema}}

Entities in the question map to the following database values:
{% for entity in entities %}
(:{{entity.label}} { name: '{{entity.name}}' })
{% endfor %}

Here are some examples: 
Example 1: For the question "List 10 titles of articles mentioning Ursula von der Leyen" and the entity list "(:Person { name: 'Ursula von der Leyen' }, (:Person { name: 'Ursula v. d. Leyn' }" the generated Cypher query should be 
"MATCH (a:Article)-[:CONTAINS]->(c:Chunk)-[:MENTIONS]->(o:Person) WHERE o.name IN ['Ursula von der Leyen', 'Ursula v. d. Leyn'] RETURN DISTINCT a.title LIMIT 10"

Example 2: For the question "How many sources mention the EU commission?" and the entity list "(:Organization { name: 'EU-Kommission' }" the generated Cypher query should be 
"MATCH (s:Source)-[:PUBLISHED]->(a:Article)-[:CONTAINS]->(c:Chunk)-[:MENTIONS]->(o:Organization) WHERE o.name IN ['EU-Kommission'] WITH DISTINCT s RETURN count(s)"

Example 3: For the question "News about France and Macron?" and the entity list "(:Location { name: 'France' }, (:Person { name: 'Emmanuel Macron' }" the generated Cypher query should be 
"MATCH (c:Chunk)-[:MENTIONS]->(o:Location) WHERE o.name = 'France' UNION MATCH (c:Chunk)-[:MENTIONS]->(o:Person) WHERE o.name = 'Emmanuel Macron' RETURN c.text LIMIT 10"

Question: {{question}}
Cypher query:"""

ANSWER_PROMPT_TEMPLATE = (
    "Answer the question below in appropriate detail, given the following context. Use only the context. "
    "If the answer is not in the context, say that you do not know."
    # "Think step by step before providing a detailed answer. "
    "The context was retrieved from the database by the following query:\n\n"
    "Query: {{query}}\n\n"
    "Context:\n{{context}}\n\n"
    "Question: {{question}}\n\n"
    "Answer: "
)

@component
class EntityRetriever:
    """A pipeline component searching for mentioned entities in the graph DB"""
    def __init__(self, db: NewsGraphClient):
        self.db = db

    @component.output_types(entities=list[Entity])
    def run(self, entities: list[Entity], per_entity_limit: int = 10):
        lookup_results = self.db.lookup_mentioned_entities(
            entities, per_entity_limit=per_entity_limit
        )
        found_entities = [
            Entity(name=result['name'], label=result['label'])
            for result in lookup_results
        ]
        return {'entities': found_entities}


@component
class CypherRetriever:
    """A pipeline component executing queries in the graph DB"""
    def __init__(self, db: NewsGraphClient):
        self.db = db

    @component.output_types(result=str)
    def run(self, queries: list[str]):
        records = []
        for query in queries:
            records.extend(self.db.query(query))
        return {
            'result': self.records_to_context(records),
            'query': ';'.join(queries)
        }
    
    def records_to_context(self, db_records: list[dict]) -> str:
        context_str = (f"\n{'='*5}\n").join(
            '/n'.join(f"{k}: {v}" for k, v in record.items())
            for record in db_records
        )
        return context_str


def setup_query_pipeline(db: NewsGraphClient|None = None):
    if db is None:
        db = NewsGraphClient()

    pipeline = Pipeline(metadata={'db': db})
    # Define pipeline components
    entity_retriever = EntityRetriever(db)
    cypher_retriever = CypherRetriever(db)
    entity_finder = EntityFinder(config.RELEVANT_LABELS)
    cypher_llm = OpenAIGenerator(
        api_key=Secret.from_env_var('GROQ_API_KEY'),
        api_base_url='https://api.groq.com/openai/v1',
        model=config.CHAT_MODEL,
        # system_prompt="Given an input question, convert it to a Cypher query. No pre-amble."
        generation_kwargs = {'max_tokens': 512, 'temperature': 0.1}
    )
    answer_llm = OpenAIGenerator(
        api_key=Secret.from_env_var('GROQ_API_KEY'),
        api_base_url='https://api.groq.com/openai/v1',
        model=config.CHAT_MODEL,  # 'mixtral-8x7b-32768',
        # system_prompt="Given an input question, convert it to a Cypher query. No pre-amble."
        generation_kwargs = {'max_tokens': 512, 'temperature': 0.1}
    )
    
    pipeline_components = (
        ('entity_finder', entity_finder),
        ('entity_retriever', entity_retriever),
        ('cypher_prompt', PromptBuilder(template=CYPHER_GENERATION_TEMPLATE)),
        ('answer_prompt', PromptBuilder(template=ANSWER_PROMPT_TEMPLATE)),
        ('cypher_llm', cypher_llm),
        ('answer_llm', answer_llm),
        # ('cypher_result', AnswerBuilder()),
        ('cypher_retriever', cypher_retriever)
    )
    # Add components to pipeline
    for component_name, component_object in pipeline_components:
        pipeline.add_component(component_name, component_object)

    # Define pipeline edges (flow of information)
    pipeline_edges = (
        ('entity_finder.entities', 'entity_retriever.entities'),
        ('entity_retriever.entities', 'cypher_prompt.entities'),
        ('cypher_prompt.prompt', 'cypher_llm.prompt'),
        ('cypher_llm.replies', 'answer_prompt.query'),
        ('cypher_llm.replies', 'cypher_retriever.queries'),
        ('cypher_retriever.result', 'answer_prompt.context'),
        ('answer_prompt.prompt', 'answer_llm.prompt')
    )
    for start, end in pipeline_edges:
        pipeline.connect(start, end)

    return pipeline



if __name__ == "__main__":
    # Steps:
    # Get input query
    # question = input('Pose a question to the LLM: ')
    # Possible questions:
    question = 'When was the article with the title "Hochrechnung zur Europawahl 2024: AfD in Ostdeutschland stärkste Kraft" published?'
    question = 'List 5 article titles about Volt'
    question = 'How many sources mention the EU parliament?'
    question = 'What do the news have to say about Olaf Scholz?'
    question = 'Was passiert in Sachsen?'
    question = 'How many sources mention BSW?'
    db = NewsGraphClient()
    pipeline = setup_query_pipeline(db)
    while (question := input('Enter a question (q to quit): ').strip()) not in ('q', 'Q'):
        result = pipeline.run(
            {
                'entity_finder': {'text': question},
                'cypher_prompt': {'schema': db.schema, 'question': question},
                'answer_prompt': {'question': question}
            }
        )
        print(result)
        print(result['answer_llm']['replies'][0])
    
    db.close()