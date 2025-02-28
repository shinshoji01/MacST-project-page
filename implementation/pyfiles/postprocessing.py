import pandas as pd
import numpy as np
from gpt import gpt_api_no_stream
import re
import collections

def get_json_result(response):
    try:
        tem = response[::-1][response[::-1].index("}"):][::-1]
    
        cumulative = ""
        extra = 1
        while extra>0:
            extra -= 1
            idxb = tem[::-1].index("{")+1
            add = tem[::-1][:idxb][::-1]
            extra += np.array([a=="}" for a in list(add[1:-1])]).sum()
            cumulative = add + cumulative
            tem = tem[::-1][idxb:][::-1]
        curlyblankets = cumulative

        # ## Preprocessing
        pattern = r"//.*?\n" # delete comment-outs
        curlyblankets = re.sub(pattern, "", curlyblankets)
        l = []
        alsonext = False
        for element in curlyblankets.split("\n"):
            if alsonext:
                alsonext = False
                continue
            if "Target Text" in element or "Backchannel" in element:
                if ":" in element[-2:]:
                    alsonext = True
                continue
            l += [element]
        curlyblankets = "\n".join(l)

        curlyblankets = curlyblankets.replace("null", '"neutral"')
        a = eval(curlyblankets)
    
    except (ValueError, SyntaxError, NameError):
        return False, None
    
    return True, a 

def CheckResultValidity(a, inputtext, normalizer):
    if len(a)==len(set(inputtext.split())):
        test = []
        for word in inputtext.split():
            exist = word in a
            if not(exist):
                normalized_word = normalizer.standardize_numbers(word)
                a_array = np.array(list(a.keys()))
                bl = normalized_word==a_array
                exist = bool(bl.sum())
                if exist:
                    a[word] = a[normalized_word]
            ## check the type of data
            if exist:
                if type(a[word])!=dict:
                    exist = False
            test += [exist]
        if np.array([test]).mean()==1:
            return True, a
    return False, None

def PostprocessTransliteration(sentence, a_list, normalizer, adds, postprocessing):
    inputtext = normalizer(sentence)
    dirs = []
    for a in a_list:
        a = {key: a[key] for key in inputtext.split()}
        dirs += [a]
    ordernames = []
    for i in range(len(dirs)):
        for key in dirs[i]:
            newlist = []
            try:
                ordername = "similarity order"
                dirs[i][key][ordername]
            except KeyError:
                ordername = "similarity_order"
            # delete duplicated words
            candidates = list(set(dirs[i][key][ordername]))
            newwords = []
            for tword in dirs[i][key][ordername]:
                if tword in candidates:
                    candidates.remove(tword)
                    newwords += [tword]
                if len(candidates)==0:
                    break
            dirs[i][key][ordername] = newwords
            for j in range(len(dirs[i][key][ordername])):
                newlist += [dirs[i][key][ordername][j]]*(len(newwords)-j)
                # newlist += [dirs[i][key][ordername][j]]
            dirs[i][key][ordername] = newlist
        ordernames += [ordername]
    data = {key: [element for i in range(len(dirs)) for element in dirs[i][key][ordernames[i]]] for key in dirs[0]}

    # Get the transliterated sentences
    arrays = []
    words = inputtext.split()
    for w, word in enumerate(words):
        if word in set([a[0].split(" ")[0] for a in list(adds.values())]):
            if word=="a":
                arrays += [postprocessing["ah"][language]]
            if word=="the":
                pro = phonemize(words[w] + " " + words[w+1], language='en-us', backend='espeak', with_stress=True).split()[0]
                for the in ["zhi", "za"]:
                    if adds[the][1][0]==pro:
                        break
                arrays += [postprocessing[the][language]]
        else:
            c = collections.Counter(data[word])
            df = pd.DataFrame(c.items(), columns=["phonemes", "count"]).sort_values("count", ascending=False).values
            arrays += [df[0,0]]

    # put period and comma
    try:
        targets = [".", ","]
        now = 0
        english_arrays = inputtext.split()
        for word in sentence.split():
        # for word in sentence.split():
            normalized = normalizer(word)
            num = len(normalized.split(" "))
            now += (num-1)
            for target in targets:
                if target in word:
                    arrays[now] += target
                    english_arrays[now] += target
            now += 1
    except IndexError:
        return None
    return " ".join(arrays)

def GetResult(client, prompt, gptmodel, inputtext, normalizer, display_print=False, error_path="gpt_error.txt"):
    repeat = True
    trial = 1
    while repeat:
        response = gpt_api_no_stream(client, prompt, model=gptmodel)[1]
        getresult, a = get_json_result(response)
        if getresult:
            valid, result = CheckResultValidity(a, inputtext, normalizer)
            if valid:
                if display_print:
                    print(f"Trial {trial}: Success!!!")
                repeat = False
            else:
                if display_print:
                    with open(error_path, 'w') as a:
                        a.write(response)
                    print(f"Trial {trial}: The result is not valid")
        else:
            if display_print:
                with open(error_path, 'w') as a:
                    a.write(response)
                print(f"Trial {trial}: Error in Converting Json Format")
        trial += 1
    return result