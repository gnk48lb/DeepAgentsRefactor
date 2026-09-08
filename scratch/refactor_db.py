import re
import os

with open('app/database.py', 'r', encoding='utf-8') as f:
    db_content = f.read()

with open('app/GraphRAG.py', 'r', encoding='utf-8') as f:
    graph_content = f.read()

# 1. Update imports in database.py
imports = """import os
import json
import logging
import config
from . import models
from . import loader as data_loader
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
from neo4j import GraphDatabase
from typing import List, Dict, Any
import shutil

logger = logging.getLogger(__name__)"""

# Replace existing imports at the top
db_content = re.sub(r'^import os.*?import shutil\n', imports + '\n', db_content, flags=re.DOTALL)

# 2. Rename collections in database.py
db_content = db_content.replace('config.COLLECTION_NAME', 'config.KNOWLEDGE_COLLECTION_NAME')
db_content = db_content.replace('coll_name = "user_memory"', 'coll_name = config.MEMORY_COLLECTION_NAME')

# 3. Modify init_av_graph_rag in database.py
db_content = db_content.replace('from .GraphRAG import DataBuilder\n        db_builder = DataBuilder()', 'db_builder = MediaDataBuilder()')

# 4. Extract DataBuilder from GraphRAG.py, rename to MediaDataBuilder
builder_match = re.search(r'class DataBuilder:.*?# --- Module 2: Query Translator ---', graph_content, re.DOTALL)
builder_code = builder_match.group(0).replace('# --- Module 2: Query Translator ---', '').strip()
builder_code = builder_code.replace('class DataBuilder:', 'class MediaDataBuilder:')
builder_code = builder_code.replace('self.collection_name = "av_works_collection"', 'self.collection_name = config.MEDIA_COLLECTION_NAME')

# 5. Get execute_hybrid_search and rerank_candidates from tools.py
with open('app/tools.py', 'r', encoding='utf-8') as f:
    tools_content = f.read()

funcs_match = re.search(r'def _execute_hybrid_search\(.*?\n\n@tool', tools_content, re.DOTALL)
funcs_code = funcs_match.group(0).replace('\n@tool', '').strip()
funcs_code = funcs_code.replace('def _execute_hybrid_search', 'def execute_hybrid_search')
funcs_code = funcs_code.replace('def _rerank_candidates', 'def rerank_candidates')

# 6. Append to database.py
db_content += '\n\n' + funcs_code + '\n\n' + builder_code + '\n'

with open('app/database.py', 'w', encoding='utf-8') as f:
    f.write(db_content)

print("database.py updated successfully")
