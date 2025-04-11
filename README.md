# Text-to-SQL Research Project

**Author:** Tahsin Jawwad  
**Supervisor:** Dr. Ramon Lawrence  
**Course:** COSC 448 – Directed Studies in Computer Science  

## Overview
This repository contains the code and experimental setups used in my Directed Studies research focused on improving Text-to-SQL conversion using GPT-3.5-turbo and modular prompting strategies. The foundational work and initial experimental setups were inspired by and adapted from the research conducted by Nascimento et al. (2024). This repository only contains the experiments that were modified and run by me. For the original experiments, use the provided GitHub repository in this README by Eduardo Nascimento. The database can also be found in that GitHub.

## Project Structure

- `experiments/`: Experimental setups, configurations, and scripts used for evaluating accuracy.
- `datasets/`: Contains key information based on connecting to the mondial database as well as the queries dataset and 
- `functions/`: Directory containing useful functions for database connections and evaluators.

## Key Functionalities

- **mschema Integration:** Improved schema representation to enhance schema linking.
- **Few-Shot Similar Pairs:** Method for boosting performance by leveraging similar natural language questions and SQL pairs.
- **C3 and DIN Integration:** Combination strategy to leverage the strengths of both C3 and DIN methodologies.
- **Embedding-based Retrieval:** Schema element retrieval using embeddings.
- **Error Validator:** Identifies and helps correct common SQL syntax errors.

## Results
The attached `Results.csv`, `Results.xlsx`, and `results.ipynb` can be used to see the results of the experiments.

## How to Run

- Install required dependencies:
```bash
pip install -r requirements.txt
```
- Configure environment variables and Oracle database connections as specified in the configuration files. Specifically, have a .env file with OPENAI_API_KEY= variable set (see `.env-example` documents in some of the repositories).
- Run experiments using provided jupyter notebooks in the `experiments` directory.

## References and Credits
- Nascimento et al. (2024) original research and repository: [text_to_sql_chatgpt_real_world](https://github.com/dudursn/text_to_sql_chatgpt_real_world)
- [XiYan-SQL](https://github.com/XGenerationLab/XiYan-SQL)
- [M-Schema](https://github.com/XGenerationLab/M-Schema)

For detailed research insights, please refer to the accompanying research report included in the repository.