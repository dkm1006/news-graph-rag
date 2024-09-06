import re
from base64 import urlsafe_b64encode
from uuid import uuid4

from huggingface_hub import HfApi


LUCENE_SPECIAL_CHARS = re.compile(r'[-+&|!(){}\[\]\^"~*?:\\]')

def generate_short_uid(prefix:str='', max_len:int=22) -> str:
    return prefix+':'+urlsafe_b64encode(uuid4().bytes).rstrip(b'=').decode('ascii')[:max_len]


def split_into_combined_sentence_chunks(text:str, min_combination_len:int=1000, len_threshold:int=1): 
    sentences = split_into_sentences(text, len_threshold=len_threshold)
    return combine_sentences(sentences, min_combination_len=min_combination_len)


def split_into_sentences(text:str, len_threshold:int=1) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"[.:;?!]", text)
        if len(sentence.strip()) > len_threshold
    ]


def combine_sentences(sentences:list[str], min_combination_len:int=1000) -> list[str]:
    temp_sentence_list = []
    combined_sentences = []
    for sentence in sentences:
        if sum(len(sentence) for sentence in temp_sentence_list) < min_combination_len:
            temp_sentence_list.append(sentence)
        else:
            combined_sentence = '.'.join(temp_sentence_list)
            combined_sentences.append(combined_sentence)
            temp_sentence_list = []

    return combined_sentences


def get_commit_hashes(repo_id):
    refs = HfApi().list_repo_refs(repo_id)
    return [branch.target_commit for branch in refs.branches]


def remove_special_chars(text: str, special_chars_pattern=LUCENE_SPECIAL_CHARS, replacement=' ') -> str:
    """Remove special characters"""
    return re.sub(special_chars_pattern, replacement, text).strip()


def generate_full_text_query(input_str: str, threshold=0.8, combine_operator='AND') -> str:
    """
    Generate a full-text search query string for a given input string.

    It processes the input string by splitting it into words and appending a
    similarity threshold (~threshold) to each word, then combines them with an operator. 
    Useful for mapping entities from user queries to database values allowing for some misspellings.
    """
    words = (word for word in remove_special_chars(input_str).split() if word)
    full_text_query = f"~{threshold} {combine_operator} ".join(words) + f"~{threshold}"
    return full_text_query
