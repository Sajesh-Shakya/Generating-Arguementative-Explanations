import pandas as pd



if __name__ == '__main__':
	vitaminc_train = pd.read_json('../data/vitaminc/train.jsonl', lines=True)
	print(vitaminc_train.head(5))