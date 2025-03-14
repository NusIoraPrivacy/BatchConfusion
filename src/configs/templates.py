old_attr_extract_template = (
    "Please extract the sensitive attributes from a question. The sensitive attributes include: \n"
    "Personal identifiable information, "
    "race, religion, ethnicity, religion, beliefs, age, "
    "sexual orientation or practices, gender identity, disability, occupation, "
    "physical or mental health (including medical conditions), vulnerable financial status, "
    "voting status, membership in a trade union, criminal record or name.\n"
    "Please strictly return a list of phrases in the question related to the sensitive attributes in the format of [\"attribute 1\", \"attribute 2\", ..., \"attribute n\"].\n"
    "If there is no sensitive attribute, return an empty list []. Try to identify as much as phrases as possible.\n"
    "Example:\n"
    "Question: "
    "A police officer carries out hundreds of traffic stops every year. He notices that the officer is equally likely to stop people of various genders, ages, and races. However, he is significantly more likely to write tickets for middle-aged white males with dark hair and eyes. What psychological framework would directly address the unconscious bias in his behavior?\n"
    "Sensitive attributes: "
    "[\"police officer\", \"middle-aged\", \"white males\", \"dark hair and eyes\"]\n"
    "Question: {question}\n"
    "Sensitive attributes: "
)

attr_extract_template = (
    "Please extract all words or phrases in the question that indicate sensitive attributes. The sensitive attributes include: \n"
    "Personal identifiable information, "
    "job, race, religion, ethnicity, religion, beliefs, age, contact information, geolocation data, residency and citizenship status, date, festival, URL, "
    "sexual orientation or practices, gender identity, third-person pronoun, disability, employment and income details, company, "
    "physical health and mental health issues, clinical signs including medical conditions and physical examination data, behavioral data, "
    "name of medicine, including specific drugs, drug classes, and general medication types (e.g., nasal decongestants, antihistamines, beta-blockers, etc),"
    "financial data, vulnerable financial status, account details, loan information, tax records, contract, trading algorithm, confidential business information, "
    "educational records, voting status, membership in a trade union, social media and digital footprint, "
    "legal consultation and case status, Law enforcement jurisdiction and procedures, consumer rights, ownership details, criminal record or name, "
    "biometric data, genetic information, legal proceedings and disputes, surveillance and monitoring data, military and security clearance.\n"
    "Your task is to strictly extract phrases that refer to these sensitive attributes. Even if a term indirectly refers to a sensitive attribute (e.g., a drug category instead of a specific medicine name), it should be included in the list.\n"
    "Please preserve the original words from the user's question to form the attribute list and do not convert full names to abbreviations or abbreviations to full names, and do not add any extra words. Retain all duplicates, regardless of form variations (e.g., burglarize, burglarized, burglarizing).\n"
    "Please strictly return a list of phrases in the format of [\"attribute 1\", \"attribute 2\", ..., \"attribute n\"].\n"
    "If there is no sensitive attribute, return an empty list []. Try to identify as much as phrases as possible.\n"
    "Example:\n"
    "Question: "
    "A 34-year-old worker (mid-age) is found to have edema, ascites, and hepatosplenomegaly. His temperature is 100\u00b0F (37.8\u00b0C), blood pressure is 97/48 mmHg, pulse is 140/min, respirations are 18/min, and oxygen saturation is 99% on room air. The man smells of alcohol and is covered in emesis. Basic laboratory values are ordered as seen below.\n\nHemoglobin: 6 g/dL\nHematocrit: 20%\nLeukocyte count: 6,500/mm^3 with normal differential\nPlatelet count: 197,000/mm^3\nReticulocyte count: 0.4%\n\nWhich of the following is associated with the most likely diagnosis?\n"
    "Sensitive attributes: "
    "[\"34-year-old\", \"worker\", \"mid-age\",\"edema\", \"ascites\", \"hepatosplenomegaly\", \"his\", \"100\u00b0F (37.8\u00b0C)\", \"97/48 mmHg\", \"140/min\", \"18/min\", \"99%\", \"man\", \"emesis\" , \"6 g/dL\", \"20%\", \"6,500/mm^3\", \"197,000/mm^3\", \"0.4%\"]\n"
    "Question: "
    "A 19-year-old girl with a history of repeated pain over the medial canthus and chronic use of nasal decongestants presents with an abrupt onset of fever with chills and rigor, diplopia on lateral gaze, moderate proptosis, chemosis, and a congested optic disc on examination. Based on these symptoms, what is the most likely diagnosis?\n"
    "Sensitive attributes: "
    "[\"19-year-old\", \"girl\", \"repeated pain\", \"medial canthus\", \"chronic use\", \"nasal decongestants\", \"fever\", \"chills\", \"rigor\", \"diplopia\", \"moderate proptosis\", \"chemosis\", \"congested optic disc\"]"
    "Question: {question}\n"
    "Sensitive attributes: "
)
# add restrictions: return the original term, do not rewrite

keyword_extract_template = (
    "Given a query, please extract the key attributes and return a list of key attributes.\n"
    "Remember to response in the format of [\"attribute 1\", \"attribute 2\", ...]."
    "Question: {question}\n"
    "Sensitive attributes: "
)

compress_template = (
    "Compress the given question to short expressions, such that you (GPT-4o) can still correctly answer the question. "
    "Please comply with the guideline below:\n"
    "1. You should remove all unneccessary or redundant information.\n"
    "2. If the question contain complex context, remember to keep all important information neccessary to answer the question.\n"
    "3. Do not use abbreviations or emojis.\n"
    "4. Compress the origin as short as you can.\n"
    "5. If the original question is already short enough, do not compress and return the original question.\n"
    "6. If the question is about fill in the blank, do not compress and return the original question.\n"
    "Please compress the following question: {question}\n"
    "Remember to return only the compressed question."
)

compress_reflect_template = (
    "The expected answer of the original question is: {true_ans}, but the previous compressed question returns the wrong answer: {wrong_ans}.\n"
    "Please evaluate your compression and identify the problem, and propose a new compression.\n"
    "Stricly provide your answer in the following format:\n"
    "#theanalysis: your analysis of the previous compression here.\n"
    "#thecompression: your new compression here."
)

compress_reflect_template_oa = (
    "The reference answer of the original question is: {ref_ans}, but the previous compressed question returns the following answer with rating {rating} (out of 10): {bad_ans}.\n"
    "Please evaluate your compression and identify the problem, and propose a new compression.\n"
    "Stricly provide your answer in the following format:\n"
    "#theanalysis: your analysis of the previous compression here.\n"
    "#thecompression: your new compression here."
)

compress_mask_template = (
    "Compress the given question to short expressions, such that you (GPT-4o) can still correctly answer the question. "
    "Please comply with the guideline below:\n"
    "1. You should remove all unneccessary or redundant information.\n"
    "2. If the question contain complex context, remember to keep all important information neccessary to answer the question.\n"
    "3. Do not use abbreviations or emojis.\n"
    "4. Compress the origin as short as you can.\n"
    "5. If the original question is already short enough, do not compress and return the original question.\n"
    "6. If the question is about fill in the blank, do not compress and return the original question.\n"
    "7. If the quesiton contain #MASK and you infer that the mask information is important, keep the #MASK appropriately in the compressed question.\n"
    "Please compress the following question: {question}\n"
    "Remember to return only the compressed question."
)

fill_compress_template = (
    "The original question is {org_question}, and the compressed question is {compression}.\n"
    "Please fill in the #MASK of the compressed question according to the original question, "
    "and obtain the unmasked compressed question. Remember to return only the unmasked compressed question."
)

local_compress_template = (
    "Please compress the give question into short expressions, "
    "such that you only remove all unneccessary or redundant information, "
    "and retain all important information neccessary to answer the question. "
    "If the original question is already short enough or is about fill in the blank, "
    "do not compress and return the original question. Question:"
)

evaluation_template = "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n[Question]\n{question}\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]"