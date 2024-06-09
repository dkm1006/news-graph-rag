import os


UID_LEN = 12
EMBEDDING_SIZE = 768
EMBEDDING_MODEL_CHECKPOINT = 'jinaai/jina-embeddings-v2-base-de'
EMBEDDING_MODEL_HASH = '5078d9924a7b3bdd9556928fcfc08b8de041bfc1'

# NOTE: For performance refer to 
#       https://docs.snowflake.com/user-guide/snowflake-cortex/llm-functions#small-models
CHAT_MODEL = 'snowflake-arctic'  # fully open source
# CHAT_MODEL = 'llama3-70b'  # better performance, open source w/ restrictions
# CHAT_MODEL = 'llama3-9b'  # smaller, open source w/ restrictions
# CHAT_MODEL = 'mistral-large'  # large european LLM, proprietary
# CHAT_MODEL = 'mixtral-8x7b'  # small european LLM, open source

SNOWFLAKE_CONNECTION_PARAMS = {
   "account": os.getenv('SNOWFLAKE_ACCOUNT'),
   "user": os.getenv('SNOWFLAKE_USER'),
   "password": os.getenv('SNOWFLAKE_PASSWORD'),
   # "role": "<your snowflake role>",  # optional
   # "warehouse": "<your snowflake warehouse>",  # optional
   # "database": "<your snowflake database>",  # optional
   # "schema": "<your snowflake schema>",  # optional
 }  
