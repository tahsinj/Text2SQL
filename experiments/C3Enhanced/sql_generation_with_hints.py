import os
import sys
from langchain.callbacks import get_openai_callback
import time
from calibration_with_hints import generate_calibration_with_hints
from generating_sql_by_type import generating_sql_by_type_prompt_maker

def generating_sql_with_hints(db, llm, question, schema_links, classification, tables=[], callback=None):
    template = generating_sql_by_type_prompt_maker(db, classification, schema_links, tables=tables)
    template = template.format(question=question)
    messages = generate_calibration_with_hints(template)
    SQL = generate(llm, messages, callback=callback)
    return SQL
    
def generate(llm, messages, callback=None):
    response = None
    while response is None:
        try:
            with get_openai_callback() as cb:
                response = llm.generate(messages)
                if response is not None and callback is not None:
                    callback({"sql_generation_din_c3": cb})
        except Exception as e:
            print(f"SQL generation API error: {e}. Retrying in 3 seconds...")
            time.sleep(3)
            pass
    try:
        SQL = response.generations[0][0].text
        SQL = SQL.split("SQL: ")[1]
    except Exception as e:
        print("SQL slicing error:", e)
        SQL = response.generations[0][0].text
    print('SQL Partial:', SQL)
    return SQL
