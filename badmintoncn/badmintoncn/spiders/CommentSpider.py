import math
import re

import scrapy
from badmintoncn.items import RacketCommentItem

class CommentSpider(scrapy.Spider):
    name = "comment_spider"
    allowed_domains = ["badmintoncn.com"]
    # 已爬取页面的 URL 集合

    def __init__(self, cookies_text: str = None, *args, **kwargs):
        super(CommentSpider, self).__init__(*args, **kwargs)
        if cookies_text is None:
            cookies_text = """
            setHits18849=y; setHits11793=y; setHits7553=y; rcKA_379b_lastvisit=1734520596; rcKA_379b_saltkey=Z2l222i1; rcKA_379b_connect_is_bind=0; rcKA_379b_myrepeat_rr=R0; userLoginJudge2036584=y; Hm_lvt_cfc948fc40dd345b6e12298c5c40ba13=1734617967,1734673628,1734698558,1734761567; HMACCOUNT=F3ACDFB3FA508B33; rcKA_379b_seccode=1175.eb729fff44658e7959; rcKA_379b_ulastactivity=1734763434%7C0; rcKA_379b_auth=34fcA0YH7oJKAhLyoNc86iNp%2BXCr41qvO2jWiU2NBbQYqWGKfxf0%2BDE7Mi6lq%2BeQRHBv3uPboxm%2F15ybD3WFZGALRTlA; cbo_auth=c1787Vm2uWEqILyTerkEZGSEJ%2F7osAkbRjugx8xgdpoqdg%2BDtQThuvXh%2BcQOiHc; oms_auth=03dfaQWHNy79F5XqaSeD4ntwxPC0farAHGL3JndlI2Ei9JlJ8BIOSovLxyWKVKQ; rcKA_379b_lastact=1734763440%09api.php%09js; Hm_lpvt_cfc948fc40dd345b6e12298c5c40ba13=1734763674
            """
        self.cookies = {}
        self.headers = {
            'user_agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
            'referer': 'https://www.badmintoncn.com/',
        }
        for item in cookies_text.strip().split(";"):
            k, _, v = item.partition("=")
            k = k.strip()
            v = v.strip()
            self.cookies[k] = v
        self.logger.debug(f"{self.cookies=}")

    def start_requests(self):
        brands = [1, 2, 22]
        pages = range(1, 11)
        for brand in brands:
            for page in pages:
                yield scrapy.Request(
                    url=f'https://www.badmintoncn.com/cbo_eq/list.php?brand={brand}&class=1&page={page}',
                    callback=self.parse_list,
                    headers=self.headers,
                )

    def parse_list(self, response):
        links = response.xpath('//a[contains(@class, "listName")]/@href').getall()
        if not links:
            self.logger.warning(f"no links found in {response.url},没找到商品链接")
        # 定义正则表达式，匹配 eid 后的数字
        else:
            pattern = r'cbo_eq/view\.php\?eid=(\d+)'
            for link in links:
                match = re.search(pattern, link)
                if match:
                    eid = match.group(1)  # 提取数字部分
                    yield scrapy.Request(
                        url=f"https://www.badmintoncn.com/cbo_eq/view_comm.php?eid={eid}&order=1",
                        callback=self.parse_comment_first,
                        headers=self.headers,
                        meta={'eid': eid}
                    )


    def parse_comment_first(self, response):
        eid = response.meta['eid']
        comment_ava_star = response.xpath(
            '//*[@id="wrap"]/div[2]/div[2]/div[2]/div[1]/table/tr/td[1]/span[1]/text()').get()
        names = response.xpath('//a[@target="_blank"]/strong/text()').getall()
        comments = response.xpath('//div[@style="margin:10px 0"]/a/text()').getall()
        img_srcs = response.xpath('//*[@class="graytext smalltext"]/img/@src').getall()
        for name, comment, img_src in zip(names, comments, img_srcs):
            matach = re.search(r'(\d+)', img_src)
            if matach:
                star = (int)(matach.group(1))
                yield RacketCommentItem(
                    eid=eid,
                    comment_id=name.strip(),
                    comment_star=star,
                    comment=comment.strip(),
                    comment_ava_star=comment_ava_star.strip()
                )

        comment_num_str = response.xpath('//*[@id="wrap"]/div[2]/div[2]/div[1]/ul/li[7]/a/text()').get()
        # 使用正则表达式匹配数字
        match = re.search(r'(\d+)', comment_num_str)
        # 如果找到数字，打印出来
        if match:
            comment_num = (int)(match.group(1))
            self.logger.info('评论数量为: %s', comment_num)
            pages = math.ceil(comment_num / 15)
            for i in range(2, pages + 1):
                yield scrapy.Request(
                    url=f'https://www.badmintoncn.com/cbo_eq/view_comm.php?eid={eid}&order=1&page={i}',
                    callback=self.parse_comment,
                    headers=self.headers,
                    cookies=self.cookies,
                    meta={'eid': eid}
                )
        else:
            self.logger.warning('No number found in string,没找到评论数量')
    def parse_comment(self, response):
        eid = response.meta['eid']
        comment_ava_star = response.xpath(
            '//*[@id="wrap"]/div[2]/div[2]/div[2]/div[1]/table/tr/td[1]/span[1]/text()').get()
        names = response.xpath('//a[@target="_blank"]/strong/text()').getall()
        comments = response.xpath('//div[@style="margin:10px 0"]/a/text()').getall()
        img_srcs = response.xpath('//*[@class="graytext smalltext"]/img/@src').getall()
        for name, comment, img_src in zip(names, comments, img_srcs):
            matach = re.search(r'(\d+)', img_src)
            if matach:
                star = (int)(matach.group(1))
                yield RacketCommentItem(
                    eid=eid,
                    comment_id=name.strip(),
                    comment_star=star,
                    comment=comment.strip(),
                    comment_ava_star=comment_ava_star.strip()
                )

