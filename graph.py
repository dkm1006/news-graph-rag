import os

import numpy as np
from neo4j import GraphDatabase, RoutingControl
from fundus.scraping.article import Article
from fundus.scraping.html import SourceInfo

# from neo4j.exceptions import DriverError, Neo4jError

import config
from schema import ArticleChunk, Iterable
from utils import generate_short_uid


# URI examples: "neo4j://localhost", "neo4j+s://xxx.databases.neo4j.io"
URI = os.getenv('DB_URL', 'neo4j://localhost:7687')
USERNAME = os.getenv('DB_USERNAME', 'neo4j')
PASSWORD = os.getenv('DB_PASSWORD', '<secret>')
AUTH = (USERNAME, PASSWORD)


class NewsGraphClient:
    def __init__(self, uri:str=URI, user:str=USERNAME, password:str=PASSWORD, database:str|None=None):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
    
    def close(self):
        self.driver.close()

    def create_article(self, article: Article) -> str:
        query = (
            "CREATE (a:Article { title: $title, publishing_date: $date, language: $language, url: $url, uid: $uid}) "
            "RETURN a.title as article_headline, a.uid"
        )
        # Use UNWIND to create Chunks
        data = {
            'uid': generate_short_uid('Article', config.UID_LEN),
            'title': article.title,
            'date': article.publishing_date,
            'language': article.lang,
            'url': article.html.responded_url
        }

        # Write transactions allow the driver to handle retries and transient errors
        record = self.driver.execute_query(
            query, database_=self.database,
            result_transformer_=lambda r: r.single(strict=True),
            **data   
        )
        return record['a.uid']
        
    def merge_article_chunks(self, article_chunks: Iterable[ArticleChunk], article_id: str):
        query = (
            "MATCH (a:Article { uid: $uid}) "
            "WITH a "
            "UNWIND $chunks as chunk "
            "CREATE (p:Chunk {text: chunk.text, category: chunk.category, section: chunk.section, position: chunk.position, uid: chunk.uid}) "
            "MERGE (a)-[:CONTAINS]->(p) "
            "RETURN a.title as article_headline, count(p) as num_paragraphs"
        )
        record = self.driver.execute_query(
            query, database_=self.database,
            result_transformer_=lambda r: r.single(strict=True),
            chunks=[chunk.to_dict(serialize=True) for chunk in article_chunks], uid=article_id
        )
        embeddings = {
            chunk.uid: chunk.embedding
            for chunk in article_chunks
        }
        record = self.set_embeddings(embeddings)
        return record

    def merge_article_authors(self, authors: Iterable[str], article_id: str):
        record = self._merge_simple_article_rel(authors, article_id, 'Person', 'AUTHORED', reverse=True)
        return record

    def merge_article_topics(self, topics: Iterable[str], article_id: str):
        record = self._merge_simple_article_rel(topics, article_id, 'Topic', 'HAS_TOPIC')
        return record

    def merge_article_source(self, source: SourceInfo, article_id: str):
        query = (
            "MATCH (a:Article { uid: $uid}) "
            "WITH a "
            "MERGE (s:Source {name: $source.publisher, type: $source.type, url: $source.url}) "
            f"ON CREATE SET s.uid = '{generate_short_uid('Source', config.UID_LEN)}' "
            "MERGE (s)-[:PUBLISHED]->(a) "
            "RETURN a.title as article_headline, s.name as source_name"
        )
        record = self.driver.execute_query(
            query, database_=self.database,
            result_transformer_=lambda r: r.single(strict=True),
            source=source.__dict__, uid=article_id
        )
        return record

    def merge_mentioned_entities(self, mentioned_entities: Iterable[dict[str, str]], article_id: str):
        query = (
            "MATCH (a:Article { uid: $uid}) "
            "WITH a "
            "UNWIND $entities as entity "
            "MERGE (e:Entity {name: entity.name}) "
            # "ON CREATE CALL apoc.create.addLabels(e, [entity.label]) YIELD e "
            "ON CREATE SET e.uid = entity.uid "
            "WITH a, e, entity "
            "MATCH (a)-[:CONTAINS]->(p:Chunk {position: entity.chunk}) "
            "MERGE (p)-[:MENTIONS]->(e) "
            "RETURN p, e"
        )
        persons, organizations, locations = [], [], []
        for entity in mentioned_entities:
            title_case_label = entity['entity'].label.title()
            data = {
                'name': entity['entity'].name,
                'label': title_case_label,
                'chunk': entity['chunk'],
                'uid': generate_short_uid(title_case_label, config.UID_LEN)
            }
            if entity['entity'].label == 'person':
                persons.append(data)
            elif entity['entity'].label == 'organization':
                organizations.append(data)
            elif entity['entity'].label == 'location':
                locations.append(data)

        records = []
        for entities, label in zip((persons, organizations, locations), ('Person', 'Organization', 'Location')):
            records.extend(self.driver.execute_query(
                query.replace('Entity', label), database_=self.database,
                result_transformer_=lambda r: r.data('p', 'e'),
                entities=entities, uid=article_id
            ))
        return records

    def set_embeddings(self, embeddings: dict[str, np.ndarray], node_type='Chunk', property_name='embedding'):
        query = (
            "UNWIND $items as item "
            f"MATCH (n:{node_type} "+"{uid: item.uid}) "
            f"CALL db.create.setNodeVectorProperty(n, '{property_name}', item.vector)"
        )
        items = [
            {'uid': uid, 'vector': embedding}
            for uid, embedding in embeddings.items()
        ]
        record = self.driver.execute_query(
            query, database_=self.database,
            result_transformer_=lambda r: r.single(),
            items=items
        )
        return record

    def setup_indexes(self):
        self.setup_performance_indexes()
        self.setup_fulltext_indexes()
        self.setup_vector_indexes()

    def setup_performance_indexes(self):
        index_list = [
            (label, 'uid', True) for label
            in ('Article', 'Chunk', 'Person', 'Organization', 'Location', 'Source', 'Topic')
        ]
        index_list.extend(
            (label, 'name', True) for label
            in ('Person', 'Organization', 'Location', 'Source', 'Topic')
        )
        index_list.extend((
            ('Article', 'url', True),
            ('Article', 'title', False),
            ('Article', 'publishing_date', False),
            ('Chunk', 'category', False)
        ))
        for label, property_name, is_unique in index_list:
            index_name = f"{label.lower()}_{property_name}_index"
            query = (
                f"CREATE {'CONSTRAINT' if is_unique else 'INDEX'} {index_name} "
                f"IF NOT EXISTS FOR (n:{label}) "
                f"{'REQUIRE' if is_unique else 'ON'} (n.{index_name}){' IS UNIQUE' if is_unique else ''}"
            )
            _ = self.driver.execute_query(
                query, database_=self.database,
                result_transformer_=lambda r: r.single()
            )
        
    def setup_fulltext_indexes(self):
        index_list = [
            (label, 'name') for label
            in ('Person', 'Organization', 'Location', 'Source', 'Topic')
        ]
        index_list.append(('Article', 'title'))
        for label, property_name in index_list:
            query = (
                f"CREATE FULLTEXT INDEX {label.lower()+property_name.title()} IF NOT EXISTS "
                f"FOR (n:{label}) ON EACH [n.{property_name}]"
            )
            _ = self.driver.execute_query(
                query, database_=self.database,
                result_transformer_=lambda r: r.single()
            )

    def setup_vector_indexes(self):
        query = (
            "CREATE VECTOR INDEX chunkEmbedding IF NOT EXISTS "
            "FOR (c:Chunk) ON c.embedding "
            "OPTIONS {indexConfig: { "
            f" `vector.dimensions`: {config.EMBEDDING_SIZE}, "
            " `vector.similarity_function`: 'cosine' }}"
        )
        _ = self.driver.execute_query(
            query, database_=self.database,
            result_transformer_=lambda r: r.single()
        )

    def _merge_simple_article_rel(self, iterable: Iterable[str], article_id: str, node_type: str, rel_type: str, prop_name='name', reverse=False):
        iterable_with_ids = [
            {'value': item, 'uid': uid} for item, uid
            in zip(iterable, iter(lambda: generate_short_uid(node_type, config.UID_LEN), None))
        ]
        query = (
            "MATCH (a:Article { uid: $uid}) "
            "WITH a "
            "UNWIND $iterable as item "
            f"MERGE (t:{node_type} "+"{"+f"{prop_name}: item.value"+"}) "
            "ON CREATE SET t.uid = item.uid "
            f"MERGE (a){'<' if reverse else ''}-[:{rel_type}]-{'' if reverse else '>'}(t) "
            "RETURN a.title as article_headline, count(t) as num_rels"
        )
        record = self.driver.execute_query(
            query, database_=self.database,
            result_transformer_=lambda r: r.single(strict=False),
            iterable=iterable_with_ids, uid=article_id
        )
        return record

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
