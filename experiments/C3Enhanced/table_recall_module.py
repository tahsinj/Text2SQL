from collections import Counter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
import time
import re

TIME_TO_SLEEP = 6

TABLE_RECALL_PROMPT = """
Given the database schema and question, perform the following actions: 
1 - Rank all the tables based on their likelihood of being used in the SQL according to the question.
2 - Ensure all tables are considered.
3 - Output a list of table names in order. Format:
[
    "table_1", "table_2", ...
]

Schema: 
{schema}

Question:
{question}

Output only the list.
"""

def robust_list_extract(text):
    try:
        list_str = re.search(r'\[.*\]', text, re.DOTALL).group(0)
        extracted_list = eval(list_str)
        if Ellipsis in extracted_list:
            extracted_list.remove(Ellipsis)
        return extracted_list
    except Exception as e:
        print(f"robust_list_extract error: {e}")
        return None

def table_recall_main(schema, tables_ori, question, llm, callback=None):
    template = TABLE_RECALL_PROMPT.format(schema=schema, question='{question}')
    prompt = PromptTemplate(template=template, input_variables=["question"])
    data_input = [{"question": question}]
    print("Table recall prompt:\n", prompt)
    tables_all = generate(llm, data_input, prompt, callback)
    table_list = table_self_consistency(tables_all, tables_ori)
    return table_list

def generate(llm, data_input, prompt, callback=None):
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    tables_all = None
    while tables_all is None:
        try:
            with get_openai_callback() as cb:
                result = llm_chain.generate(data_input)
                print("Tables result:", result)
                tables_all = get_tables_response(result)
                if tables_all is not None and callback is not None:
                    callback({"table_recall": cb})
                else:
                    time.sleep(TIME_TO_SLEEP)
        except Exception as e:
            print(f"API error in table recall: {e}. Retrying after {TIME_TO_SLEEP} seconds...")
            time.sleep(TIME_TO_SLEEP)
    return tables_all

def get_tables_response(responses):
    all_tables = []
    for table_response in responses.generations[0]:
        raw_table = table_response.text
        extracted = robust_list_extract(raw_table)
        if extracted is None:
            print("list error in table recall")
            return None
        all_tables.append(extracted)
    return all_tables

def table_self_consistency(tables_all, tables_ori):
    tables_sc = []
    for tables in tables_all:
        tables_exist = []
        for table in tables:
            if table.lower() in tables_ori:
                tables_exist.append(table.lower())
                if len(tables_exist) == 4:
                    break
        tables_sc.append(tables_exist)
    from collections import Counter
    counts = Counter(tuple(sorted(lst)) for lst in tables_sc)
    if counts:
        most_list, count = counts.most_common(1)[0]
        for table_list in tables_sc:
            if sorted(table_list) == list(most_list):
                return table_list
    return []
