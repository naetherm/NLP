from transformers import BartTokenizer, BartForConditionalGeneration 

model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

# Source: https://www.varsitytutors.com/praxis_reading-help/long-passages-200-400-words
text = """The next point is the color of the mature caterpillars, some of which are brown. This probably makes the caterpillar even more conspicuous among the green leaves than would otherwise be the case. Let us see, then, whether the habits of the insect will throw any light upon the riddle. What would you do if you were a big caterpillar? Why, like most other defenseless creatures, you would feed by night, and lie concealed by day. So do these caterpillars. When the morning light comes, they creep down the stem of the food plant, and lie concealed among the thick herbage and dry sticks and leaves, near the ground, and it is obvious that under such circumstances the brown color really becomes a protection. It might indeed be argued that the caterpillars, having become brown, concealed themselves on the ground, and that we were reversing the state of things. But this is not so, because, while we may say as a general rule that large caterpillars feed by night and lie concealed by day, it is by no means always the case that they are brown; some of them still retaining the green color. We may then conclude that the habit of concealing themselves by day came first, and that the brown color is a later adaptation."""

input = tokenizer([text], max_length=1024, return_tensors='pt')

sum_ids = model.generate(
    input['input_ids'], 
    num_beams=4, 
    max_length=100, 
    early_stopping=True)

summarized_text = ([tokenizer.decode(i, skip_special_tokens=True, clean_up_tokenization_spaces=False) for i in sum_ids])

print(summarized_text)
