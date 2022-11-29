import pandas as pd

TARGET_FILE_NAME = input("나눌 파일 이름 입력 : ")
DATA_FILE_NAME = "train.tsv"
VAL_FILE_NAME = "valid.tsv"

# df1 = datasX.iloc[:, :72]
# df2 = datasX.iloc[:, 72:] iloc 함수 쓰자
df = pd.read_csv(TARGET_FILE_NAME, delimiter='\t')

number_of_data = len(df)

data_len = number_of_data * 8 // 10

data_df = df.iloc[:, :data_len]
val_df = df.iloc[:, data_len:]

data_df.to_csv(DATA_FILE_NAME, index=False, header=True, sep="\t")
val_df.to_csv(VAL_FILE_NAME , index=False, header=True, sep="\t")