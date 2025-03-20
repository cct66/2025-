import csv


# 读取CSV文件
def read_csv(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头
        data = [row for row in reader]
    return data


# 找出2024年首次出现的国家
def find_new_countries(data):
    years = {}
    for row in data:
        year = int(row[-1])  # 获取年份
        noc = row[1]  # 获取国家/地区代码
        if year not in years:
            years[year] = set()
        years[year].add(noc)

    # 按年份排序
    sorted_years = sorted(years.keys())

    # 找出2024年首次出现的国家
    new_countries_2024 = set()
    for year in sorted_years:
        if year == 2024:
            previous_years = set()
            for prev_year in sorted_years:
                if prev_year < 2024:
                    previous_years.update(years[prev_year])
            new_countries_2024 = years[2024] - previous_years
            break

    return new_countries_2024


# 获取2024年新出现国家的奖牌数
def get_new_countries_medals(data, new_countries):
    new_countries_medals = {}
    for row in data:
        if int(row[-1]) == 2024 and row[1] in new_countries:
            noc = row[1]
            gold = int(row[2])
            silver = int(row[3])
            bronze = int(row[4])
            total = gold + silver + bronze
            new_countries_medals[noc] = {
                'Gold': gold,
                'Silver': silver,
                'Bronze': bronze,
                'Total': total
            }
    return new_countries_medals


# 主函数
def main():
    file_path = 'summerOly_medal_counts.csv'  # CSV文件路径
    data = read_csv(file_path)
    new_countries_2024 = find_new_countries(data)
    new_countries_medals = get_new_countries_medals(data, new_countries_2024)

    print("2024年首次出现在夏季奥运会奖牌榜中的国家及其奖牌数：")
    for noc, medals in new_countries_medals.items():
        print(f"{noc}: Gold {medals['Gold']} Silver {medals['Silver']} Bronze {medals['Bronze']} Total {medals['Total']}")


if __name__ == "__main__":
    main()