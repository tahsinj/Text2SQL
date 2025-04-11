from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine, text
from prompts.validator_prompt import *
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
from din_parameter import TIME_TO_SLEEP
import time

def validate_sql(db, mschema, llm_din_fix, question, query, callback=None):
    validated, error = validate(query)
    print(validated)
    if validated:
        return query
    validation_prompt = validate_sql_prompt_builder(db, mschema)
    count = 0
    while not validated and count < 3:
        data_input = [{'question': question, 'query': query, 'error': error}]
        query = generate(llm_din_fix, data_input, validation_prompt, callback=callback)

def validate(query):
    try:
        db_engine = create_engine("oracle+oracledb://MONDIAL_GPT:TextDB123@localhost:1522/?service_name=XEPDB1")
        with db_engine.connect() as connection:
            connection.execute(text("EXPLAIN PLAN FOR " + query))  # Dry-run the query
        return True, None  # Query is valid
    except SQLAlchemyError as e:
        print(f"SQL Validation Error: {e}")
        return False, e  # Query is invalid
    
def generate(llm, data_input, prompt, callback=None):
    
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response = None
    while response is None:
        try:
            with get_openai_callback() as cb:
                response = llm_chain.generate(data_input)
                if response is not None:
                    if callback is not None:
                        callback({"self_correction":cb})
        except:
            time.sleep(TIME_TO_SLEEP)
            pass
   
    debugged_SQL = response.generations[0][0].text.replace("\n", " ")
    SQL = "SELECT " + debugged_SQL
    return SQL

def validate_sql_prompt_builder(db, mschema):
    schema_basic_prompt = mschema.to_mschema()
    fk_basic_prompt = db.get_foreign_keys_basic_prompt()
    pk_basic_prompt = db.get_primary_keys_basic_prompt()
    template = validator_prompt.format(schema=schema_basic_prompt, 
                                             primary_keys=pk_basic_prompt,
                                             foreign_keys=fk_basic_prompt,
                                             dialect=db.get_dialect(),
                                             question='{question}',
                                             query='{query}',
                                             error='{error}')
    prompt = PromptTemplate(template=template, input_variables=['question', 'query'])
    return prompt