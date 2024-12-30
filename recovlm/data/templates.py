"""Templates"""

chat_template = (
  "{% for message in messages %}"
  "{% if loop.first and messages[0]['role'] != 'system' %}"
  "{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}"
  "{% endif %}"
  "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
  "{% endfor %}"
  "{% if add_generation_prompt %}"
  "{{ '<|im_start|>assistant\n' }}"
  "{% endif %}"
)

chat_template_with_generation_tag = (
  "{% for message in messages %}"
  "{% if loop.first and messages[0]['role'] != 'system' %}"
  "{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}"
  "{% endif %}"
  "{% if (message['role'] in ['system', 'user']) %}"
  "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
  "{% else %}"
  "{{ '<|im_start|>assistant\n' }}"
  "{% generation %}"
  "{{ message['content'] + '<|im_end|>\n' }}"
  "{% endgeneration %}"
  "{% endif %}"
  "{% endfor %}"
)

chat_template_response_only = (
  "{% for message in messages %}{{message['content'] + '<|im_end|>'}}{% endfor %}"
)

def get_template(name):
  if name == "chat_template_with_generation_tag":
    return chat_template_with_generation_tag
  elif name == "chat_template_response_only":
    return chat_template_response_only
  else:
    return chat_template
