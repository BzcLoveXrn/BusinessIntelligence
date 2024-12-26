import math
import re

import scrapy
from badmintoncn.items import TrainDataItem

class TrainDataSpider(scrapy.Spider):
    name = 'train_data_spider'
    allowed_domains = ['badmintoncn.com']
    def __init__(self, cookies_text: str = None, *args, **kwargs):
        super(TrainDataSpider, self).__init__(*args, **kwargs)
        if cookies_text is None:
            cookies_text = """
            setHits9706=y; setHits9748=y; setHits11536=y; setHits21593=y; setHits20961=y; setHits17114=y; setHits17699=y; setHits20874=y; setHits7717=y; setHits18979=y; setHits8674=y; setHits13503=y; setHits21856=y; setHits20950=y; setHits21917=y; rcKA_379b_lastvisit=1734520596; rcKA_379b_saltkey=Z2l222i1; rcKA_379b_connect_is_bind=0; rcKA_379b_auth=5b6dn5bPSP48ezvui6u31iI19mwWrLFs%2FBnSWfAga%2FUIbqmhh8VxNrM5rFTyWB4ASrax8KKgcXDCquCAcFERBJ8Hyidg; cbo_auth=cf71kxcgPeXh3ncg1aHKj1SW%2BV%2BHPVFsGIOPr2xDcfbSBrk0rOzcSpVEBjKggZg; oms_auth=e4999mAM9V9L58%2Fhe0eqrmk1qEIFHnGoN7au1%2BW2Pr0XU5q7NXegp0dTZ%2FmCpF8; userLoginJudge2036584=y; Hm_lvt_cfc948fc40dd345b6e12298c5c40ba13=1734761567,1734838939,1734862399,1735037860; HMACCOUNT=F3ACDFB3FA508B33; rcKA_379b_ulastactivity=1735037870%7C0; bbsbottomcookie=yes; Hm_lpvt_cfc948fc40dd345b6e12298c5c40ba13=1735040345; rcKA_379b_lastact=1735040393%09api.php%09js
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
        brands = [13, 516, 7]
        pages = range(1, 5)
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
        comments = response.xpath('//div[@style="margin:10px 0"]/a/text()').getall()
        img_srcs = response.xpath('//*[@class="graytext smalltext"]/img/@src').getall()
        for  comment, img_src in zip(comments, img_srcs):
            matach = re.search(r'(\d+)', img_src)
            if matach:
                star = (int)(matach.group(1))
                yield TrainDataItem(
                    eid=eid,
                    comment_star=star,
                    comment=comment.strip(),
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
        comments = response.xpath('//div[@style="margin:10px 0"]/a/text()').getall()
        img_srcs = response.xpath('//*[@class="graytext smalltext"]/img/@src').getall()
        for comment, img_src in zip(comments, img_srcs):
            matach = re.search(r'(\d+)', img_src)
            if matach:
                star = (int)(matach.group(1))
                yield TrainDataItem(
                    eid=eid,
                    comment_star=star,
                    comment=comment.strip(),
                )
