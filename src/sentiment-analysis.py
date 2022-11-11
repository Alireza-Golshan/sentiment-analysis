from transformers import AutoModelForSequenceClassification, AutoTokenizer
from constants import auth_key, token_size
from requests import post
from pathlib import Path
from natsort import natsorted
from PyPDF2 import PdfReader


def load_data_sources():
    directory = '../interviews'
    files = Path(directory).glob('*')
    files = natsorted(files, key=str)
    return files


def normalize_data(files):
    data_dic = {}
    for interview in files:
        reader = PdfReader("../interviews/" + interview.name)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"

        final_text = ""
        for line in text.splitlines():
            if '?' in line:
                final_text += "\nanswer " + "\n"
            else:
                final_text += line

        dkey = interview.name[interview.name.rfind("e-") + 2:interview.name.rfind(".")]
        data_dic[dkey] = final_text

    return data_dic


def get_datasource():
    files = load_data_sources()
    data_sources = normalize_data(files)
    return data_sources


def prepare_model():
    model = AutoModelForSequenceClassification.from_pretrained("rabiaqayyum/autotrain-mental-health-analysis-752423172",
                                                               use_auth_token=False)
    tokenizer = AutoTokenizer.from_pretrained("rabiaqayyum/autotrain-mental-health-analysis-752423172",
                                              use_auth_token=False)
    inputs = tokenizer("text", return_tensors="pt")
    outputs = model(**inputs)
    return outputs


def send_request(data):
    json_data = {'inputs': data}
    response = post(
        'https://api-inference.huggingface.co/models/rabiaqayyum/autotrain-mental-health-analysis-752423172',
        headers={'Authorization': 'Bearer ' + auth_key, }, json=json_data)
    return response.text


data_source = get_datasource()

for key, value in data_source.items():
    print("=========================================================================================")
    print("\n" + key)
    answers = value.split("answer")
    answers.pop(0)
    ans = 1
    for answer in answers:
        print("metrics for answer number " + str(ans))
        if len(answer) > token_size:
            answer = answer[0:512]
        res = send_request(answer)
        print(res + "\n")
        ans += 1
    print("=========================================================================================\n")
