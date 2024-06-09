from math import ceil

import fundus
import fundus.scraping.article

from embedding import embed_sentences
from graph import NewsGraphClient
from ner import EntityFinder
from schema import ArticleChunk, ArticleChunkCategory, Iterable
from utils import split_into_combined_sentence_chunks, generate_short_uid


MAX_PARAGRAPH_LEN = 1100

    
def main():
    publishers = fundus.PublisherCollection.de
    crawler = fundus.Crawler(*publishers)
    max_articles = 2
    articles = crawler.crawl(max_articles=max_articles)
    db = NewsGraphClient()
    for article in articles:
        # title, body, plaintext
        # body contains a summary and sections, each section a headline, paragraphs
        # lang, publishing_date, topics, authors
        # Article: contains metadata - links to sections, Sections contain paragraphs
        article_id = db.create_article(article=article)  # includes metadata and title
        print(article_id)
        article_chunks = get_chunks_from_article_body(article)
        embeddings = embed_sentences(*(chunk.text for chunk in article_chunks))
        for chunk, embedding in zip(article_chunks, embeddings):
            chunk.embedding = embedding
        
        _ = db.merge_article_chunks(article_chunks, article_id)
        print(_)
        topics = article.topics  # name only (Entity Topic)
        _ = db.merge_article_topics(topics, article_id)
        print(_)
        source = article.html.source_info  #publisher, type, url (Entity Source)
        _ = db.merge_article_source(source, article_id)
        print(_)
        authors = article.authors or [source.publisher]  # name only (Entity Author, if empty take generic Source?)
        _ = db.merge_article_authors(authors, article_id)
        print(_)
        entity_finder = EntityFinder(labels=['person', 'organization', 'location'])
        mentioned_entities = (
            {
                'entity': entity,
                'section': chunk.section,
                'chunk': chunk_idx
            }
            for chunk_idx, chunk in enumerate(article_chunks)
            for entity in entity_finder.find_entities_in_texts(chunk.text)
        )
        _ = db.merge_mentioned_entities(mentioned_entities, article_id)
        for r in _:
            print(r)


def get_chunks_from_article_body(article: fundus.scraping.article.Article) -> list[ArticleChunk]:
    article_chunks = list(
        chunk_text_sequence(article.body.summary, category=ArticleChunkCategory.SUMMARY, section_idx=0)
    )
    for i, section in enumerate(article.body.sections, start=1):
        headline_chunks = chunk_text_sequence(section.headline, category=ArticleChunkCategory.HEADLINE, section_idx=i)
        article_chunks.extend(headline_chunks)
        paragraph_chunks = chunk_text_sequence(section.paragraphs, category=ArticleChunkCategory.PARAGRAPH, section_idx=i)
        article_chunks.extend(paragraph_chunks)

    for i, chunk in enumerate(article_chunks):
        chunk.position = i

    return article_chunks


def chunk_text_sequence(text_sequence: Iterable[str], category: ArticleChunkCategory, section_idx: int):
    return (
        ArticleChunk(text=text, category=category, section=section_idx)
        for text in ensure_max_len_of_texts(text_sequence)
    )


def ensure_max_len_of_texts(text_sequence: Iterable[str], max_len:int=MAX_PARAGRAPH_LEN):
    """Splits texts into smaller chunks if their length is above a threshold"""
    for text in text_sequence:
        if len(text) < max_len:
            yield text
        else:
            # In order to split evenly
            min_combination_len = int(max_len / ceil(len(text) / max_len))
            yield from split_into_combined_sentence_chunks(text, min_combination_len)


if __name__ == '__main__':
    main()