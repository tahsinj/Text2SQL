from langchain.schema import HumanMessage, AIMessage, SystemMessage

C3_HINTS_PROMPT = [
    SystemMessage(
        content="""
                You are now an excellent SQL writer. First, I'll give you some tips and examples, and I need you to 
                remember the tips and do not make the same mistakes.
                """
    ),
    HumanMessage(
        content="""
            Tips 1:
            Question: Which A has most number of B?
            Gold SQL: select A from B group by A order by count (*) desc fetch first 1 rows only;
            Notice that the Gold SQL doesn't select COUNT(*) because the question only wants to know the A and
            the number should be used only in the ORDER BY clause.
            """
    ),
    AIMessage(
        content="""
            Thank you for the tip! I'll keep in mind that when the question only asks for a certain field, I should not
            include COUNT(*) in the SELECT clause, but instead use it in ORDER BY.
            """
    ),
    HumanMessage(
        content="""
            Tips 2:
            Don't use "IN", "OR", or "LEFT JOIN" as they might cause extra results; use "INTERSECT" or "EXCEPT"
            instead, and remember to use "DISTINCT" or "FETCH FIRST" when necessary.
            For example:
            Question: Who are the A who have been nominated for both B award and C award?
            Gold SQL: select A from X where award = 'B' intersect select A from X where award = 'C';
            """
    ),
    AIMessage(
        content="""
            Thank you for the tip! I'll remember to use "INTERSECT" or "EXCEPT" instead of "IN", "NOT IN", or
            "LEFT JOIN" when matching records across tables, and use "DISTINCT" or "FETCH FIRST" to avoid duplicates.
            """
    )
]

def generate_calibration_with_hints(clear_prompting):
    messages = C3_HINTS_PROMPT.copy()
    input_user = HumanMessage(content=clear_prompting)
    messages.append(input_user)
    return [messages]
