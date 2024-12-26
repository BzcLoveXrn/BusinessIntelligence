import numpy as np
import pandas as pd
def filters(data_all, brand=None, weight=None, price_start=0, price_end=1000000):
    data = data_all.copy()
    """根据品牌、重量和价格过滤数据"""
    if brand:
        data = data[data['brand'].isin(brand)]  # 使用 isin 进行品牌过滤
    if weight:
        # 遍历每个用户指定的重量并检查是否为子串
        weight_condition = data['racket_weight'].apply(lambda x: any(w in x for w in weight))
        data = data[weight_condition]
    data = data[(data['price_new'] >= price_start) & (data['price_new'] <= price_end)]
    return data


def get_recommendations(data,brand=None, weight=None, price_start=0, price_end=1000000,labels=None):
    filtered_data = filters(data, brand, weight, price_start, price_end)
    if labels is None:
        return filtered_data.head(10)
    else:
        distances = []
        for _, row in filtered_data.iterrows():
            distance = 0
            for label,(value,weight) in labels.items():
                diff = row[label] - value
                distance += (diff ** 2) * weight
            distances.append(np.sqrt(distance))
        filtered_data['distance'] = distances
        return filtered_data.sort_values('distance').head(10)
data = pd.read_csv("../DataAnalysis/data/data.csv")
# brand = ['尤尼克斯 YONEX', '威克多 VICTOR']
# weight=['3U', '4U']
# labels={'杀球': [0.39, 5]}
#
# hohho=get_recommendations(data=data,brand=brand,weight=weight,labels=labels,price_start=300,price_end=500)
# print(hohho)


