import pandas as pd



file_list = ['', '', '', '']
frames = []


for i in file_list:
    frames.append(pd.read_csv(i, delimiter='\t'))

# 어떻게 설정할지 고민중...
# 이거 하고 자동으로 파일 나눠주는거
# https://stackoverflow.com/questions/41624241/pandas-split-dataframe-into-two-dataframes-at-a-specific-row
# df1 = datasX.iloc[:, :72]
# df2 = datasX.iloc[:, 72:] iloc 함수 쓰자
result = pd.concat(frames,ignore_index=True)

result.to_csv('test.tsv', index=False, header=True, sep="\t")
print(result)