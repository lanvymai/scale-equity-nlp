import re
generated_texts = ['User:<image>What is this a picture of? Provide only the option\'s letter from the given choice.OPTIONS:\nA.The image in El Sauce, Nicaragua\nB.Sámi noaidi with his drum\nC.A tableau presenting figures from various cultures described as "shamans" in Western academic literature.\nD.Khiamniungans dance outside their morung at the Hornbill Festival\nAssistant: Answer: C']

match = re.search(r'Assistant: (?:Answer: )?(\w)', generated_texts[0])


# If a match is found, extract the letter (i.e., the answer)
if match:
    prediction = match.group(1)
    print(prediction)