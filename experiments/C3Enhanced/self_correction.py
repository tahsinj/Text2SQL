from prompts.self_correction_prompt import *
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
from parameter import TIME_TO_SLEEP
import mschema
import time
import re

def fix_common_errors(sql):
    sql = re.sub(r"'\d{1,2}/\d{1,2}/\d{4}'", lambda m: fix_date_formats(m.group(0)), sql)
    return sql

# Helper: reusing the fix_date_formats function from din_generating_sql_by_type.py
def fix_date_formats(date_str):
    def repl(match):
        parts = match.group(0).replace("'", "").split("/")
        if len(parts) == 3:
            month = parts[0].zfill(2)
            day = parts[1].zfill(2)
            year = parts[2]
            return f"DATE '{year}-{month}-{day}'"
        return match.group(0)
    return re.sub(r"'\d{1,2}/\d{1,2}/\d{4}'", repl, date_str)

def generate(llm, data_input, prompt, callback=None):
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response = None
    while response is None:
        try:
            with get_openai_callback() as cb:
                response = llm_chain.generate(data_input)
                if response is not None and callback is not None:
                    callback({"self_correction": cb})
        except Exception as e:
            time.sleep(TIME_TO_SLEEP)
            pass
    debugged_SQL = response.generations[0][0].text.replace("\n", " ")
    SQL = "SELECT " + debugged_SQL
    SQL = fix_common_errors(SQL)
    return SQL

def self_correction_prompt_maker(db, mschema):
    schema_basic_prompt = mschema.to_mschema()
    fk_basic_prompt = db.get_foreign_keys_basic_prompt()
    pk_basic_prompt = db.get_primary_keys_basic_prompt()
    template = self_correction_prompt.format(schema=schema_basic_prompt, 
                                             primary_keys=pk_basic_prompt,
                                             foreign_keys=fk_basic_prompt,
                                             dialect=db.get_dialect(),
                                             question='{question}',
                                             query='{query}')
    prompt = PromptTemplate(template=template, input_variables=['question', 'query'])
    return prompt

def self_correction_module(db, mschema, llm, question, sql, callback=None):
    prompt = self_correction_prompt_maker(db, mschema)
    data_input = [{'question': question, 'query': sql}]
    sql_query_fix = generate(llm, data_input, prompt, callback=callback)
    return sql_query_fix
