from collections import Counter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
import time
import json
import re

TIME_TO_SLEEP = 6

COLUMN_RECALL_PROMPT = """
Given the database tables and question, perform the following actions: 
1 - Rank the columns in each table based on their likelihood of being used in the SQL. Columns that match the question words or are part of a foreign key should be ranked higher.
2 - Output a JSON object with each table’s columns ordered by relevance. Format:
{
    "table_1": ["column_1", "column_2", ...],
    "table_2": ["column_1", "column_2", ...],
    ...
}

Schema: 
{schema}
{foreign_keys}

Question:
### {question}

Output only the JSON object.
"""

def robust_json_extract(text):
    try:
        json_str = re.search(r'\{.*\}', text, re.DOTALL).group(0)
        return json.loads(json_str)
    except Exception as e:
        print(f"robust_json_extract error: {e}")
        return None

def column_recall_main(schema, tabs_cols_ori, question, llm, foreign_keys_prompt, callback=None):
    prompt = PromptTemplate(template=COLUMN_RECALL_PROMPT, input_variables=["schema", "foreign_keys", "question"])
    fk_prompt = "Foreign keys:\n" + foreign_keys_prompt if len(foreign_keys_prompt) > 0 else foreign_keys_prompt
    data_input = [{"schema": schema, "foreign_keys": fk_prompt, "question": question}]
    print("Column recall prompt:\n", prompt)
    tabs_cols_all = generate(llm, data_input, prompt, callback=callback)
    if tabs_cols_all is None:
        return None
    return column_self_consistency(tabs_cols_all, tabs_cols_ori)

def generate(llm, data_input, prompt, callback=None):
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    tabs_cols_all = None
    attempts = 1
    while tabs_cols_all is None and attempts <= 3:
        try:
            with get_openai_callback() as cb:
                result = llm_chain.generate(data_input)
                print("Column recall result:", result)
                tabs_cols_all = get_tables_column_response(result)
                if tabs_cols_all is not None and callback is not None:
                    callback({"column_recall": cb})
                else:
                    time.sleep(TIME_TO_SLEEP)
        except Exception as e:
            print(f"API error on column recall attempt {attempts}: {e}")
            time.sleep(TIME_TO_SLEEP)
        finally:
            print(f"Column recall attempt: {attempts}")
            attempts += 1
    return tabs_cols_all

def get_tables_column_response(responses):
    tabs_cols_all = []
    for tabs_cols_response in responses.generations[0]:
        raw_tab_col = tabs_cols_response.text
        extracted = robust_json_extract(raw_tab_col)
        if extracted is None:
            print("list error column recall")
            return None
        tabs_cols_all.append(extracted)
    return tabs_cols_all

def column_self_consistency(tabs_cols_all, tabs_cols_ori):
    candidates = {key: [] for key in tabs_cols_ori}
    results = {}
    for tabs_cols in tabs_cols_all:
        for key, value in tabs_cols.items():
            if key in tabs_cols_ori:
                candidates[key].append(value)
    for tab, cols_all in candidates.items():
        cols_ori = [item.lower() for item in tabs_cols_ori[tab]]
        cols_sc = []
        for cols in cols_all:
            cols_exist = [col for col in cols if col.lower() in cols_ori]
            if cols_exist:
                cols_sc.append(cols_exist)
        if cols_sc:
            cols_add = [col for sublist in cols_sc for col in sublist]
            from collections import Counter
            counter = Counter(cols_add)
            most_common_cols = [col for col, count in counter.most_common(5)]
            results[tab] = most_common_cols
        else:
            results[tab] = []
    return results
