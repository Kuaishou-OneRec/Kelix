from sglang import function, system, user, assistant, gen, \
    set_default_backend, RuntimeEndpoint

# @function
# def multi_turn_question(s, question_1, question_2):
#     s += system("You are a helpful assistant.")
#     s += user(question_1)
#     s += assistant(gen("answer_1", max_tokens=256))
#     s += user(question_2)
#     s += assistant(gen("answer_2", max_tokens=256))

set_default_backend(RuntimeEndpoint("http://127.0.0.1:50000"))

# state = multi_turn_question.run(
#     question_1="What is the capital of the United States?",
#     question_2="List two local attractions.",
# )

@function
def text_qa(s, question):
    s += "Q: " + question + "\n"
    s += "A:" + gen("answer", stop="\n")

states = text_qa.run_batch(
    [
        {"question": "What is the capital of the United Kingdom?"},
    ] * 100000,
    progress_bar=True
)

print(states)
