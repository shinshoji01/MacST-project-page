import glob
import numpy as np
import collections
import pandas as pd
from phonemizer import phonemize

def gpt_api_no_stream(
    client,
    prompt: str, 
    model="gpt-4o",
    reset_messages: bool = True,
    response_only: bool = True
):
    """
    ------------
    Examples
    ------------
    
    try:
        response = gpt_api_no_stream(prompt, model=model)[1]
    except AuthenticationError:
        continue
    if "OpenAI API error" in response:
        print(f"{response}")
    else:
        np.save(savepath, response)
    """
    
    if "gpt-3.5" in model:
        model = "gpt-3.5-turbo-1106"
    elif "gpt-4omini" in model:
        model = "gpt-4o-mini-2024-07-18"
    elif "gpt-4o" in model:
        model = "gpt-4o-2024-11-20"
    elif "gpt-o1mini" in model:
        model = "o1-mini-2024-09-12"
    messages = [{'role': 'user','content': prompt},]

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        completion = dict(completion)
        msg = None
        choices = completion.get('choices', None)
        if choices:
            msg = choices[0].message.content
        else:
            msg = completion.message.content
    except Exception as err:
        return (False, f'OpenAI API error: {err}')
    if reset_messages:
        messages.pop(-1)
    else:
        # add text of response to messages
        messages.append({
            'role': choices[0].message.role,
            'content': choices[0].message.content
        })
    if response_only:
        return True, msg
    else:
        return True, messages


def GetLLMPrompt(sentence, language, phonemized=None):
    words = sentence.split()
    if type(phonemized)!=list:
        phonemized = [phonemize(word, language='en-us', backend='espeak', with_stress=True).split()[0] for word in words]
    shfflephonemized = phonemized

    start = f"""Can you provide me with three {language} words to represent the phoneme sequences delimited by triple backticks. 
For example, in Japanese, "Trail (tɹˈeɪl)" is expected to have Japanese representation of "トレイル"; where "'" in phonemes represents the stress point of the word. 
Here, your task is to provide me with three {language} words that can replace the phoneme senquences, delimited by triple backticks.
Please focus on phonetically similar characters instead of similar characters in terms of the meaning.
The expected output should be in JSON format. 
You can first list three possible choices of the words and then re-order them in order of the similarity of the pronunciation. 
The following is the example in Hindi language.
{{
  "I": {{
    "phonemes": "ˈaɪ",
    "choices": ["आई", "ऐ", "आई"],
    "similarity order": ["आई", "ऐ", "आई"]
  }},
  "love": {{
    "phonemes": "lˈʌv",
    "choices": ["लव", "लव", "लव"],
    "similarity order": ["लव", "लव", "लव"]
  }},
  "you": {{
    "phonemes": "juː",
    "choices": ["यू", "यू", "यू"],
    "similarity order": ["यू", "यू", "यू"]
  }},
}}
```
"""
    for p, ph in enumerate(shfflephonemized):
        start += f"{words[p]}: {ph}\n"
    start = start[:-1]
    start += f"""
```
Again, the responses should be in a JSON format and sort them in order of the similarity to each phoneme sequence.
{{
"""
    for p, ph in enumerate(shfflephonemized):
        start += f"""  "{words[p]}": {{
"phonemes": "{ph}",
"choices": [`1st choices of {language} characters`, `2nd choices of {language} characters`, `3rd choices of {language} characters`],
"similarity order": [`1st most similar {language} characters`, `2nd most similar {language} characters`, `3rd most similar {language} characters`],
}},\n"""
    start = start[:] + "}"
    return start

