import requests
from pathlib import Path
from PyPDF2 import PdfReader
from matplotlib import pyplot as plt
from natsort import natsorted
from requests.adapters import HTTPAdapter
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from urllib3.util.retry import Retry
from constants import auth_key, token_size


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
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    response = session.post(
        'https://api-inference.huggingface.co/models/rabiaqayyum/autotrain-mental-health-analysis-752423172',
        headers={'Authorization': 'Bearer ' + auth_key, }, json=json_data)
    return response.json()


data_source = get_datasource()

person = 1
for key, value in data_source.items():
    print("=========================================================================================")
    print("\n" + key)
    depression = bipolar = bpd = anxiety = mentalhealth = schizophrenia = autism = 0
    answers = value.split("answer")
    answers.pop(0)
    ans = 1
    for answer in answers:
        print("metrics for answer number " + str(ans))
        if len(answer) > token_size:
            answer = answer[0:512]
        res = send_request(answer)
        results = res[0]
        print(results)
        for result in results:
            if 'label' in result and result['label'] == 'depression':
                depression += result['score']
            if 'label' in result and result['label'] == 'bipolar':
                bipolar += result['score']
            if 'label' in result and result['label'] == 'BPD':
                bpd += result['score']
            if 'label' in result and result['label'] == 'Anxiety':
                anxiety += result['score']
            if 'label' in result and result['label'] == 'mentalhealth':
                mentalhealth += result['score']
            if 'label' in result and result['label'] == 'schizophrenia':
                schizophrenia += result['score']
            if 'label' in result and result['label'] == 'autism':
                autism += result['score']

        ans += 1
    depression = float("{0:.2f}".format(depression / (ans - 1)))
    bipolar = float("{0:.2f}".format(bipolar / (ans - 1)))
    bpd = float("{0:.2f}".format(bpd / (ans - 1)))
    anxiety = float("{0:.2f}".format(anxiety / (ans - 1)))
    mentalhealth = float("{0:.2f}".format(mentalhealth / (ans - 1)))
    schizophrenia = float("{0:.2f}".format(schizophrenia / (ans - 1)))
    autism = float("{0:.2f}".format(autism / (ans - 1)))

    plt.style.use('ggplot')
    x = ['depression', 'bipolar', 'BPD', 'Anxiety', 'mentalhealth', 'schizophrenia', 'autism']
    data = [depression, bipolar, bpd, anxiety, mentalhealth, schizophrenia, autism]
    x_pos = [i for i, _ in enumerate(x)]
    plt.bar(x_pos, data, color='blue')
    plt.xlabel("Mental Health")
    plt.ylabel("Estimate")
    plt.title("Results for person " + str(person))
    plt.xticks(x_pos, x)
    plt.gca().set_ylim([0.00, 0.50])
    plt.show()
    person += 1

print("=========================================================================================\n")
