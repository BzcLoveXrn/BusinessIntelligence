# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

# items.py

import scrapy


class RacketItem(scrapy.Item):
    eid = scrapy.Field()  # 羽毛球拍编号
    price_new = scrapy.Field()  # 一手价格
    price_old = scrapy.Field()  # 二手价格
    name_usuall = scrapy.Field()  # 羽毛球拍名称,大众叫法
    name_official = scrapy.Field()  # 羽毛球拍名称,官方叫法
    brand = scrapy.Field()  # 品牌
    series = scrapy.Field()  # 系列
    launch_date = scrapy.Field()  # 发布日期
    racket_weight = scrapy.Field()  # 羽毛球拍重量
    racket_length = scrapy.Field()  # 羽毛球拍长度
    grip_thickness = scrapy.Field()  # 手柄粗细
    stringing_tension = scrapy.Field()  # 拉线磅数




class RacketCommentItem(scrapy.Item):
    eid = scrapy.Field()  # 羽毛球拍编号
    comment_id = scrapy.Field()  # 评论用户名
    comment_ava_star = scrapy.Field()  # 综合评分
    comment_star = scrapy.Field()  # 评论星级
    comment = scrapy.Field()  # 评论内容

class TrainDataItem(scrapy.Item):
    eid = scrapy.Field()  # 羽毛球拍编号
    comment_star = scrapy.Field()  # 评论星级
    comment = scrapy.Field()  # 评论内容
