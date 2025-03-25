import json
import nltk
from nltk.stem import PorterStemmer
from transformers import pipeline
from nltk.corpus import stopwords
import time
# Ensure stopwords are downloaded

# nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load a pre-trained NER model from Hugging Face
print('++++++++++')
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
print('1')
# Sample input JSON
json_data = '''[
{
    "Question": "I was wondering if a pain management office is acting illegally/did an illegal action.. I was discharged as a patient from a pain management office after them telling me that a previous pain management specialist I saw administered a steroid shot wrong and I told them in the portal that I spoke to lawyers for advice but no lawsuit/case was created. It was maybe 1-2 months after I was discharged that I no longer have access to my patient portal with them. Every time I try to login I enter my credentials, wait a few seconds, and then I get re-directed back to the original screen where I have various options to login. I know I can speak to the office directly and ask them about what specifically is going on, talk to other lawyers if this is a violation of my rights, etc. but I was just wondering if anyone on this site would know if this action is in fact illegal.",
    "Answer": "In Kentucky, your situation raises questions about patient rights and medical records access. If you were discharged from a pain management office and subsequently lost access to your patient portal, it's important to understand your rights regarding medical records. Under the Health Insurance Portability and Accountability Act (HIPAA), you have the right to access your own medical records. A healthcare provider cannot deny you access to your records, even if you were discharged from their practice. The issue with the patient portal could be a technical glitch or a deliberate action, but either way, it needs to be addressed. You should contact the pain management office directly to inquire about the issue with the patient portal. They are obligated to provide you with a way to access your medical records, which may include providing paper copies or access through a different electronic means. If the office is unresponsive or refuses to provide access to your records, consider speaking to a lawyer who is knowledgeable in health care law. They can guide you on how to enforce your rights to access your medical records and address any potential legal violations by the healthcare provider. Remember, your right to access your medical records is protected by law, and any obstacles to this access should be promptly addressed.",
    "private attributes": [
        "pain management", "offices", "discharged", "patient", "pain management", "office", "pain management",
        "specialist", "steroid shot", "lawyers", "lawsuit/case", "1-2 months", "discharged", "patient",
        "portal", "login", "credentials", "login", "office", "lawyers", "a violation of my rights", "illegal", "year", "jobs", "condition"
    ]
}
]'''

data = json.loads(json_data)

def filter_private_attributes(attributes):
    # ps = PorterStemmer()
    # attributes = [ps.stem(word) for word in attributes]
    # attributes = set(attributes)  # Remove duplicates
    meaningful_words = set()

    # Run NER model on all attributes
    ner_results = ner_pipeline(" ".join(attributes))

    # Extract named entities with high confidence
    for item in ner_results:
        if item["score"] > 0.85:
            meaningful_words.add(item["word"])

    # Apply stopword filtering from NLTK
    filtered_attributes = [
        word for word in attributes
        if word.lower() not in stop_words
        and (word in meaningful_words or len(word) > 3)  # Keep important words & long words
    ]

    return filtered_attributes

# Process each entry
for item in data:
    item["filtered_private_attributes"] = filter_private_attributes(item["private attributes"])

# Print the output
print(json.dumps(data, indent=4))
