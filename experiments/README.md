## Experiments

In this repository, C3, DIN, DIN-C3, SequentialDatabaseChain, SQLAgents, SQLCoder, and SQLQueryChain have gone through minimal changes to only replicate experiments using the latest versions of some libraries. The original versions of these experiments can be found here: [text_to_sql_chatgpt_real_world](https://github.com/dudursn/text_to_sql_chatgpt_real_world).

The C3+, mschema, and SimilarPairs are experiments derived from other provided experiments that are fully working and have been tested. The results can be found in the root of this repository in `Results.xlsx`.

The C3Enhanced has not been tested and was in the works to implement slight improvements, such as feeding error messages of the SQLValidator, robust schema extraction, fallback to embeddings based retrieval, and candidate SQL generation.