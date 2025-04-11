validator_prompt = """
#### For the given question, running the given Oracle SQL query forms the following error: {error}. Fix the SQL Query using the provided tables, columns, foreign keys, and primary keys and return only the fixed Oracle SQL Query. Focus on only fixing the query based on the error.

{schema}
Foreign_keys = {foreign_keys}
Primary_keys = {primary_keys}
#### Question: {question}
#### {dialect} SQL QUERY
{query}
#### {dialect} FIXED SQL QUERY
SELECT
"""