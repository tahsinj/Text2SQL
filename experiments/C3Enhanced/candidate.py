import numpy as np
from langchain.chains import LLMChain
from parameter import TIME_TO_SLEEP
from langchain.callbacks import get_openai_callback
import time

prompt = """You are a highly proficient Oracle SQL generator. Your task is to generate {num_candidates} distinct candidate SQL queries for the following natural language question. Do not provide any commentary or explanation; output only the SQL queries. Each candidate must be preceded by the string "SQL: " (without quotes).

Question: "{question}"

Schema: {schema}

Instructions:
1. Analyze the question carefully and decide on an appropriate SQL strategy.
2. Use correct Oracle syntax (e.g., FETCH FIRST ... ROWS ONLY for limits, proper Oracle date formatting, and proper handling of joins).
3. If multiple approaches are possible (e.g., different join strategies or subquery constructions), output each distinct formulation.
4. Ensure that each candidate represents a complete SQL statement that can be independently executed.
5. Your output must include exactly {num_candidates} candidate SQL queries, each on a separate line starting with "SQL: ".

Now, generate the candidate SQL queries.

"""

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
    return response


def generate_candidate_sql(llm, question, schema, num_candidates=10, callback=None):
    data_input = [{"question": question, "num_candidates": num_candidates, "schema": schema}]

    response = generate(llm, data_input, prompt)
    candidates = []
    for candidate in response.generations[0]:
        try:
            sql_text = candidate.text.split("SQL: ")[1].strip()
        except Exception as e:
            print(f"Candidate extraction error: {e}")
            sql_text = candidate.text.strip()
        candidates.append(sql_text)
    if callback:
        callback({"candidate_sqls": candidates})
    return candidates

def execute_candidates(db, candidates):
    outputs = []
    for sql in candidates:
        try:
            df = db.run_sql_query(sql)
            outputs.append(df)
        except Exception as e:
            print(f"Execution error for candidate: {e}")
            outputs.append(None)
    return outputs

def compute_output_consistency(candidate_outputs):
    n = len(candidate_outputs)
    scores = [0] * n
    for i in range(n):
        df_i = candidate_outputs[i]
        if df_i is None:
            continue
        for j in range(n):
            if i == j:
                continue
            df_j = candidate_outputs[j]
            if df_j is None:
                continue
            try:
                if df_i.equals(df_j): ## could be different
                    scores[i] += 1
            except Exception as e:
                continue
    return scores

def select_best_candidate(candidates, candidate_outputs):
    scores = compute_output_consistency(candidate_outputs)
    if len(scores) == 0:
        return None, None
    best_index = np.argmax(scores)
    best_candidate = candidates[best_index]
    return best_candidate, best_index

def candidate_sql_generation(llm, question, prompt, db, num_candidates=10, callback=None):
    print("Generating candidate SQL queries...")
    candidates = generate_candidate_sql(llm, question, prompt, num_candidates=num_candidates, callback=callback)
    
    print("Executing candidate SQL queries...")
    candidate_outputs = execute_candidates(db, candidates)
    
    best_candidate, best_index = select_best_candidate(candidates, candidate_outputs)
    print(f"Selected candidate index: {best_index} with highest consistency.")
    if callback:
        callback({"best_candidate": best_candidate, "all_candidates": candidates, "outputs": candidate_outputs})
    return best_candidate, candidates, candidate_outputs
