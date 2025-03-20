import chardet
import pandas as pd
# 检测文件编码
with open('../summerOly_programs.csv', 'rb') as f:
    result = chardet.detect(f.read())

# 打印检测到的编码
print(result['encoding'])

# 使用检测到的编码读取文件
programs_df = pd.read_csv('../summerOly_programs.csv', encoding=result['encoding'])
