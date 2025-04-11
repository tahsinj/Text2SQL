import os
import sys
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
import time
from mschema.schema_engine import SchemaEngine
from nltk.tokenize import word_tokenize
from fuzzywuzzy import fuzz
from sqlalchemy import create_engine, text
import nltk
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from mschema.schema_engine import MSchema

from table_recall_module import table_recall_main
from column_recall_module import column_recall_main
from prompts_template.schema_linking_template import SCHEMA_LINKING_PROMPT

TIME_TO_SLEEP = 4

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def generate(llm, data_input, prompt, callback=None):
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response = None
    while response is None:
        try:
            with get_openai_callback() as cb:
                response = llm_chain.generate(data_input)
                if response is not None:
                    if callback is not None:
                        callback({"schema_linking":cb})
        except:
            time.sleep(TIME_TO_SLEEP)
            pass
    try:
        schema_links = response.generations[0][0].text
        schema_links = schema_links.split("Schema_links: ")[1]
    except:
        print("Slicing error for the schema_linking module")
        schema_links = "[]"
        
    return schema_links


def schema_linking_din_c3(question, db, mschema, llm_c3, llm_din, add_fk=False, callback=None):
    mschema_str = mschema.to_mschema()
    
    # table_schema = db.get_schema_openai_prompt()
    tables_ori = db.get_table_names()
    table_list = table_recall_main(mschema_str, tables_ori, question, llm_c3, callback=callback)

    specific_tables_schema = db.get_schema_openai_prompt(table_list)
    specific_tables_mschema = filter_mschema_by_tables(mschema, table_list)
    specific_tables_mschema_str = specific_tables_mschema.to_mschema()

    tables_cols_ori = db.get_schema_json(table_list)
    foreign_keys_prompt = ""
    if add_fk:
        foreign_keys_prompt = db.get_foreign_keys_openai_prompt(table_list)
    
    print(specific_tables_mschema_str)

    schema_result = column_recall_main(specific_tables_mschema_str, tables_cols_ori, question, llm_c3, foreign_keys_prompt, callback=callback)
    
    if schema_result is not None:
        schema_result_prompt = get_schema_to_clear_prompt(schema_result, foreign_keys_prompt)
    else:
        schema_result_prompt = db.get_schema_openai_prompt(table_list) + foreign_keys_prompt
            
    template = SCHEMA_LINKING_PROMPT.format(schema=schema_result_prompt, question="{question}")
    prompt = PromptTemplate(template=template, input_variables=["question"])
    data_input = [{"question": question}]
    
    schema_linking_result = generate(llm_din, data_input, prompt, callback=callback)
    return schema_linking_result, table_list

def get_schema_to_clear_prompt(schema_result, foreign_keys_prompt):
    schema_result_pompt = ""
    for tbl, columns in schema_result.items():
        schema_result_pompt += f"# {tbl} ({', '.join([c for c in columns])})\n"
    
    schema_result_pompt += foreign_keys_prompt
        
    return schema_result_pompt

def get_table_representation(mschema, table):
    table_info = mschema.tables.get(table, {})
    # Get column names
    columns = list(table_info.get("fields", {}).keys())
    # Collect examples from each field (if available)
    examples = []
    for field, details in table_info.get("fields", {}).items():
        field_examples = details.get("examples", [])
        if field_examples:
            # You can choose to include only a few examples per field
            examples.extend(field_examples[:2])
    # Combine table name, columns, and examples; lower-case for consistency.
    representation = f"{table.lower()} " + " ".join([col.lower() for col in columns])
    if examples:
        representation += " " + " ".join([ex.lower() for ex in examples])
    return representation

def filter_tables_by_embedding(mschema, question, top_k=5, threshold=None):
    # Extract all table names from mschema
    all_tables = list(mschema.tables.keys())
    
    # Lowercase the question for consistency
    question = question.lower()
    query_embedding = embedding_model.encode(question, normalize_embeddings=True)
    table_scores = {}
    
    for table in all_tables:
        table_repr = get_table_representation(mschema, table)
        table_embedding = embedding_model.encode(table_repr, normalize_embeddings=True)
        score = float(cosine_similarity([query_embedding], [table_embedding])[0][0])
        table_scores[table] = score
    
    if threshold is not None:
        selected_tables = [table for table, score in table_scores.items() if score >= threshold]
    else:
        sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
        selected_tables = [table for table, score in sorted_tables[:top_k]]
    
    # Create a new MSchema instance containing only the selected tables
    from mschema.schema_engine import MSchema
    filtered_mschema = MSchema(db_id=mschema.db_id, schema=mschema.schema)
    filtered_mschema.tables = {table: details for table, details in mschema.tables.items() if table in selected_tables}
    filtered_mschema.foreign_keys = [
        fk for fk in mschema.foreign_keys if fk[0] in selected_tables and fk[3] in selected_tables
    ]
    return filtered_mschema


def filter_mschema_by_tables(mschema, table_list):
    try:
        # Check if mschema is an instance of MSchema
        if hasattr(mschema, 'tables'):
            filtered_tables = {
                table: details 
                for table, details in mschema.tables.items() 
                if table in table_list
            }
            # Create a new MSchema instance with filtered tables
            from mschema.schema_engine import MSchema  # ensure MSchema is imported
            mschema_filtered = MSchema(db_id=mschema.db_id, schema=mschema.schema)
            mschema_filtered.tables = filtered_tables
            # Optionally filter foreign keys to include only those between filtered tables
            mschema_filtered.foreign_keys = [
                fk for fk in mschema.foreign_keys 
                if fk[0] in table_list and fk[3] in table_list
            ]
            return mschema_filtered
        else:
            # If mschema is already a dict, do the filtering directly.
            if "tables" in mschema:
                filtered_tables = {
                    table: details 
                    for table, details in mschema["tables"].items() 
                    if table in table_list
                }
                mschema["tables"] = filtered_tables
            return mschema
    except Exception as e:
        print("Error filtering mschema:", e)
        return mschema