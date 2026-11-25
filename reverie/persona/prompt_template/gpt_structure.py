"""
Author: Joon Sung Park (joonspk@stanford.edu)
File: gpt_structure.py
Description: Wrapper functions for calling OpenAI APIs.

MODIFIED for this re-implementation (see NOTICE):
  * The original called the retired `openai.Completion` (text-davinci-002/003)
    and `openai.ChatCompletion` (gpt-3.5-turbo / gpt-4) endpoints with the
    pre-1.0 SDK. All calls now go through llm_gateway (current SDK, one model
    setting, cache, cost tracking, budget, mock mode).
  * Completion-style prompts are sent unchanged to a chat model with an
    instruction to continue the text; the original temperature, max_tokens,
    top_p and stop sequences are kept.
  * Errors no longer turn into the string "ChatGPT ERROR": transient errors
    are retried, fatal ones stop the run (FatalLLMError).
  * The function names, signatures, prompt wrapping and retry loops are the
    authors' own.
"""
import json
import os

from llm_gateway import gateway, FatalLLMError

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(os.path.dirname(_HERE))


def temp_sleep(seconds=0.1):
  pass  # the original slept 0.1 s before each call to avoid rate limits


def ChatGPT_single_request(prompt):
  return gateway().chat(prompt)


def GPT4_request(prompt):
  return gateway().chat(prompt, model=gateway().strong_model)


def ChatGPT_request(prompt):
  return gateway().chat(prompt)


def _json_part(text):
  """The JSON object in a model answer. Current models often wrap it in a
  ```json code fence; the authors cut only the text after the last '}', so a
  fence in front broke json.loads on every try."""
  text = (text or "").strip()
  start, end = text.find("{"), text.rfind("}") + 1
  return text[start:end] if start != -1 and end > start else text


def _fail_safe(fail_safe_response):
  """All tries failed. The authors pass a fail-safe answer for exactly this
  case but returned False, which their callers turn into None and a crash
  ('NoneType' object is not subscriptable). Use the fail-safe and log it."""
  from llm_gateway import gateway
  gw = gateway()
  key = "fail_safe_used"
  gw.stats[key] = gw.stats.get(key, 0) + 1
  print(f"FAIL-SAFE used for {gw.current_template}: {fail_safe_response!r}")
  return fail_safe_response


def GPT4_safe_generate_response(prompt,
                                example_output,
                                special_instruction,
                                repeat=3,
                                fail_safe_response="error",
                                func_validate=None,
                                func_clean_up=None,
                                verbose=False):
  prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'
  for i in range(repeat):
    try:
      curr_gpt_response = _json_part(GPT4_request(prompt))
      curr_gpt_response = json.loads(curr_gpt_response)["output"]
      # Current models often answer {"output": 6} instead of {"output": "6"};
      # the authors' validators expect text, so give them text.
      if isinstance(curr_gpt_response, (int, float)) and not isinstance(curr_gpt_response, bool):
        curr_gpt_response = str(curr_gpt_response)
      if func_validate(curr_gpt_response, prompt=prompt):
        return func_clean_up(curr_gpt_response, prompt=prompt)
    except Exception:
      pass
  return _fail_safe(fail_safe_response)


def ChatGPT_safe_generate_response(prompt,
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False):
  prompt = '"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'
  if verbose:
    print("CHAT GPT PROMPT")
    print(prompt)
  for i in range(repeat):
    try:
      curr_gpt_response = _json_part(ChatGPT_request(prompt))
      curr_gpt_response = json.loads(curr_gpt_response)["output"]
      # Current models often answer {"output": 6} instead of {"output": "6"};
      # the authors' validators expect text, so give them text.
      if isinstance(curr_gpt_response, (int, float)) and not isinstance(curr_gpt_response, bool):
        curr_gpt_response = str(curr_gpt_response)
      if func_validate(curr_gpt_response, prompt=prompt):
        return func_clean_up(curr_gpt_response, prompt=prompt)
    except Exception:
      pass
  return _fail_safe(fail_safe_response)


def ChatGPT_safe_generate_response_OLD(prompt,
                                       repeat=3,
                                       fail_safe_response="error",
                                       func_validate=None,
                                       func_clean_up=None,
                                       verbose=False):
  for i in range(repeat):
    try:
      curr_gpt_response = ChatGPT_request(prompt).strip()
      if func_validate(curr_gpt_response, prompt=prompt):
        return func_clean_up(curr_gpt_response, prompt=prompt)
    except Exception:
      pass
  print("FAIL SAFE TRIGGERED")
  return fail_safe_response


def GPT_request(prompt, gpt_parameter):
  """Completion-style request (originally text-davinci-00x)."""
  return gateway().chat(prompt,
                        completion=True,
                        temperature=gpt_parameter.get("temperature"),
                        max_tokens=gpt_parameter.get("max_tokens"),
                        top_p=gpt_parameter.get("top_p"),
                        stop=gpt_parameter.get("stop"))


def generate_prompt(curr_input, prompt_lib_file):
  """
  Takes in the current input and the path to a prompt file, and replaces
  every !<INPUT n>! with the n-th input. (Authors' function; it now also
  records which template is in use, for cost reports and mock mode.)
  """
  if type(curr_input) == type("string"):
    curr_input = [curr_input]
  curr_input = [str(i) for i in curr_input]

  path = prompt_lib_file
  if not os.path.isabs(path):
    path = os.path.join(_BACKEND, prompt_lib_file)
  gateway().current_template = prompt_lib_file
  with open(path, "r", encoding="utf-8") as f:
    prompt = f.read()
  for count, i in enumerate(curr_input):
    prompt = prompt.replace(f"!<INPUT {count}>!", i)
  if "<commentblockmarker>###</commentblockmarker>" in prompt:
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
  return prompt.strip()


def safe_generate_response(prompt,
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False):
  if verbose:
    print(prompt)
  for i in range(repeat):
    curr_gpt_response = GPT_request(prompt, gpt_parameter)
    if func_validate(curr_gpt_response, prompt=prompt):
      return func_clean_up(curr_gpt_response, prompt=prompt)
  return fail_safe_response


def get_embedding(text, model=None):
  text = text.replace("\n", " ")
  if not text:
    text = "this is blank"
  return gateway().embed(text)
