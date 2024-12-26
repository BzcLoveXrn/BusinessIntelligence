# app.py
from flask import Flask, render_template, request, jsonify
from API import get_recommendations
import pandas as pd
import json

app = Flask(__name__)

data = pd.read_csv("../DataAnalysis/data/data.csv")
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    # 获取表单数据
    brands = request.form.getlist('brands[]')
    weights = request.form.getlist('weights[]')
    price_start = request.form.get('price_start', type=int)
    price_end = request.form.get('price_end', type=int)
    labels = {}
    LABELS = ['入门', '高端', '性价比', '暴力', '进攻', '杀球', '控制', '头重', '连贯', '速度', '中杆硬', '中杆软',
              '糖水', '颜值', '拉吊']
    for label in LABELS:
        t_value = request.form.get(f'{label}_值', type=float)
        t_weight = request.form.get(f'{label}_权重', type=int)
        if t_weight != 0:
            labels[label] = [t_value, t_weight]
    if not brands:
        brands=None
    if not weights:
        weights=None
    if price_start is None:
        price_start = 0
    if price_end is None:
        price_end = 1000000
    print(brands, weights, price_start, price_end, labels)
    recommendations = get_recommendations(data=data,brand=brands, weight=weights, price_start=price_start, price_end=price_end, labels=labels)
    return render_template('recommendations.html', recommendations=recommendations[:10])





if __name__ == '__main__':
    app.run(debug=True)
