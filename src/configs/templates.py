temp_attr_extract_template = (
    "Please extract all words or phrases in the question that indicate sensitive attributes.\n"
    "The sensitive attributes include: Personal identifiable information, "
    "job, race, religion, ethnicity, religion, beliefs, age, contact information, geolocation data, residency and citizenship status, date, festival, URL, "
    "sexual orientation or practices, gender identity, third-person pronoun, disability, employment and income details, company, "
    "physical health and mental health issues, clinical signs including medical conditions and physical examination data, behavioral data, "
    "name of medicine, including specific drugs, drug classes, and general medication types (e.g., nasal decongestants, antihistamines, beta-blockers, etc), "
    "financial data, vulnerable financial status, account details, loan information, tax records, contract, trading algorithm, confidential business information, "
    "educational records, voting status, membership in a trade union, social media and digital footprint, "
    "legal consultation and case status, Law enforcement jurisdiction and procedures, consumer rights, ownership details, criminal record or name, "
    "biometric data, genetic information, legal proceedings and disputes, surveillance and monitoring data, military and security clearance.\n"
    "Your task is to strictly extract phrases that refer to these sensitive attributes. The list of sensitive attributes provided is not exhaustive — if you encounter any word or phrase that could reasonably be considered sensitive information under privacy laws or common data protection standards (like GDPR, HIPAA, or similar frameworks), include it in the list. Prioritize any data that could identify a person, describe their personal circumstances, or reveal confidential, medical, financial, or legal information. Please do not includes too general words with weak sensitive information (e.g., \"office\", \"work\", \"will\", \"report\")\n"
    "Please preserve the original words from the user's question to form the attribute list and do not convert full names to abbreviations or abbreviations to full names, and do not add any extra words. Retain all duplicates, regardless of form variations (e.g., burglarize, burglarized, burglarizing).\n"
    "Please strictly return a list of phrases in the format of [\"attribute 1\", \"attribute 2\", ..., \"attribute n\"]. If there is no sensitive attribute, return an empty list []. Try to identify as much as phrases as possible.\n"
    "Question: {question}\n"
    "Sensitive attributes: "
)


attr_extract_template = (
    "Please extract all words or phrases in the question that indicate sensitive attributes.\n"
    "The sensitive attributes include: Personal identifiable information, "
    "job, race, religion, ethnicity, religion, beliefs, age, contact information, geolocation data, residency and citizenship status, date, festival, URL, "
    "sexual orientation or practices, gender identity, third-person pronoun, disability, employment and income details, company, "
    "physical health and mental health issues, clinical signs including medical conditions and physical examination data, behavioral data, "
    "name of medicine, including specific drugs, drug classes, and general medication types (e.g., nasal decongestants, antihistamines, beta-blockers, etc), "
    "financial data, vulnerable financial status, account details, loan information, tax records, contract, trading algorithm, confidential business information, "
    "educational records, voting status, membership in a trade union, social media and digital footprint, "
    "legal consultation and case status, Law enforcement jurisdiction and procedures, consumer rights, ownership details, criminal record or name, "
    "biometric data, genetic information, legal proceedings and disputes, surveillance and monitoring data, military and security clearance.\n"
    "Your task is to strictly extract phrases that refer to these sensitive attributes. Even if a term indirectly refers to a sensitive attribute (e.g., a drug category instead of a specific medicine name), it should be included in the list. The list of sensitive attributes provided is not exhaustive — if you encounter any word or phrase that could reasonably be considered sensitive information under privacy laws or common data protection standards (like GDPR, HIPAA, or similar frameworks), include it in the list. Prioritize any data that could identify a person, describe their personal circumstances, or reveal confidential, medical, financial, or legal information.\n"
    "Please preserve the original words from the user's question to form the attribute list and do not convert full names to abbreviations or abbreviations to full names, and do not add any extra words. Retain all duplicates, regardless of form variations (e.g., burglarize, burglarized, burglarizing).\n"
    "Please strictly return a list of phrases in the format of [\"attribute 1\", \"attribute 2\", ..., \"attribute n\"]. If there is no sensitive attribute, return an empty list []. Try to identify as much as phrases as possible.\n"
    "Question: {question}\n"
    "Sensitive attributes: "
)

attr_generated_template_one_context = (  
    "You are an AI designed to generate accurate, contextually relevant, and trustworthy fake sensitive attributes."  
    "Your task is to generate one list of five fake sensitive attributes for each original attribute provided in filtered private attributes based on the context of the 'question'. \n"
    "The generated attributes must meet the following criteria:\n"
    " 1. Numerical Handling:"
    "     - If the original attribute is primarily numerical (e.g., percentages, durations, counts), generate fake numerical attributes that have broader variance to reduce guessability."
    "     - Avoid clustering numbers closely around the original value."
    "     - Avoid adding unnecessary descriptive context (e.g., avoid turning '50%' into '55% of alternative components'). Instead, generate standalone numerical fakes like '20%', '90%', or '10%'."
    "     - Do not introduce numerical values if the original attribute is non-numerical in any form. (e.g. aviod turning 'internship period' into '3 months')\n"
    " 2. Length Consistency: Ensure that the generated fake attributes have a similar length (in words or characters) to the original attribute to maintain fluency and reduce guessability."
    " 3. Sound Sensitive and Relaiable:"
    "     - The fake attributes should resemble private, confidential, or sensitive concepts, particularly in legal, medical, or financial domains."
    "     - Generated fake attributes actually exist in real-world literature.\n"
    " 4. No Rephrasings or Simple Synonyms: Avoid generating superficial rephrasings or synonyms (e.g., do not turn '80 years' into 'eight decades'):"
    "     - Create attributes that keep contextual depth and believability."
    "     - Do not generate fakes by slightly modifying words."
    "     - The genearted word should under the same category of the original attribute but with strictly distinct meanings.\n"
    " 5. Diversity and Independence:"
    "     - Generate independent fake attributes for each prompt and do not reference or rely on fake attributes generated for the other prompt. Avoid generating fake attributes that are too similar to each other."
    "     - Ensure that the fake attributes are diverse and do not share common themes or patterns."
    "     - if the input private attributes are dependent based on the context, generate fake attributes that are dependent on the context as well.\n"
    " 6. Contextual Fluency:"
    "     - Make sure that the fake attributes fit naturally and coherently within the sentence structure for each context."
    "     - The input attributes may contain some abbreviation. Please generate the fake attributes based on the full name of the input attributes."
    "     - The fake attributes can be in abbreviation or full name.\n"
    "Please strictly return one list of fake attributes in the type of list. apart from the list, you should not add any word.\n"
    "The list must be structured as follows:"
    "    - Each row(sublist, element of list) must contain exactly 6 elements."
    "    - The first element must be the original attribute."
    "    - The next 5 elements must be unique, fake attributes."
    "    - Output format for each list is: [[\"original attribute 1\", \"fake attribute 1_1\", \"fake attribute 1_2\", \"fake attribute 1_3\", \"fake attribute 1_4\", \"fake attribute 1_5\"],[\"original attribute 2\", \"fake attribute 2_1\", \"fake attribute 2_2\", \"fake attribute 2_3\", \"fake attribute 2_4\", \"fake attribute 2_5\"],..., [\"original attribute m\", \"fake attribute m_1\", \"fake attribute m_2\", \"fake attribute m_3\", \"fake attribute m_4\", \"fake attribute m_5\"]]\n"
    "The input is structured as follows:\n"
    "Filtered Private Attributes: {filtered_private_attributes}\n"
    "question: {question}\n"  
    "fake attributes for question:"  
)  


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