# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html
import json
import os

# useful for handling different item types with a single interface
from itemadapter import ItemAdapter

import pandas as pd

class XlsxExportPipeline:
    def __init__(self):
        # 用来存储每种类型的 item 数据
        self.info_data = []
        self.comment_data = []
        self.train_data = []


    def process_item(self, item, spider):
        if item is None:
            return item
        # 根据不同的 item 类型，分别处理并保存到对应的列表中
        if spider.name == 'basic_info_spider':
            self.info_data.append(dict(item))  # 保存基本信息数据
            spider.logger.info("基本信息加入元祖")
        elif spider.name == 'comment_spider':
            self.comment_data.append(dict(item))  # 保存评论数据
            spider.logger.info("评论加入元祖")
        elif spider.name == 'train_data_spider':
            self.train_data.append(dict(item))  # 保存训练数据
            spider.logger.info("训练数据加入列表")
        return item

    def close_spider(self, spider):
        # 当爬虫关闭时，将数据保存为不同的 Excel 文件
        if self.info_data:
            file_path = 'info_data.csv'
            df = pd.DataFrame(self.info_data)
            if os.path.exists(file_path):
                df.to_csv(file_path, index=False, header=False, mode='a')
                print("Data has been appended to output.csv")
            else:
                df.to_csv(file_path, index=False, header=True)
                print("Data has been written to output.csv")


        if self.comment_data:
            file_path = 'comment_data.csv'
            df = pd.DataFrame(self.comment_data)
            if os.path.exists(file_path):
                df.to_csv(file_path, index=False, header=False, mode='a')
                print("Data has been appended to output.csv")
            else:
                df.to_csv(file_path, index=False, header=True)
                print("Data has been written to output.csv")

        if self.train_data:
            file_path = 'train_data.csv'
            df = pd.DataFrame(self.train_data)
            if os.path.exists(file_path):
                df.to_csv(file_path, index=False, header=False, mode='a')
                print("Data has been appended to train_data.csv")
            else:
                df.to_csv(file_path, index=False, header=True)
                print("Data has been written to train_data.csv")


class VisitedUrlsJsonPipeline:

    def __init__(self):
        # 初始化时，尝试读取已保存的 URL 集合
        self.visited_urls = set()
        self.file_name = 'visited_urls.json'

        if os.path.exists(self.file_name):
            # 如果文件存在，加载已爬取的 URL 集合
            with open(self.file_name, 'r') as f:
                self.visited_urls = set(json.load(f))  # 将 JSON 数据转回 set

    def process_item(self, item, spider):
        # 处理每个抓取到的 item
        # 如果该 URL 已经爬取过，则跳过
        url = item.get('url')
        if url in self.visited_urls:
            return None  # 直接跳过
        self.visited_urls.add(url)
        return item

    def close_spider(self, spider):
        # 在爬虫结束时，保存 URL 集合到文件
        with open(self.file_name, 'w') as f:
            json.dump(list(self.visited_urls), f)  # 将 set 转回 list 存储到 JSON
        spider.logger.info(f"Visited URLs have been saved to {self.file_name}")
