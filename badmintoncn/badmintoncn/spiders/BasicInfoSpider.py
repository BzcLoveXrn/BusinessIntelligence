import re

import scrapy

from badmintoncn.items import RacketItem
class BasicInfoSpider(scrapy.Spider):
    name = "basic_info_spider"
    allowed_domains = ["badmintoncn.com"]

    def __init__(self, cookies_text: str = None, *args, **kwargs):
        super(BasicInfoSpider, self).__init__(*args, **kwargs)
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
                        url=f"https://www.badmintoncn.com/cbo_eq/view_specs.php?eid={eid}",
                        callback=self.parse_basic_info,
                        headers=self.headers,
                        meta={'eid': eid}
                    )


    def parse_basic_info(self, response):
        eid = response.meta['eid']
        name_usuall = response.xpath('//*[@id="wrap"]/div[1]/a[5]/text()').get()
        name_official = response.xpath('//*[@id="wrap"]/div[2]/div[2]/div[2]/div[1]/table/tr[6]/td[2]/text()').get()
        brand = response.xpath('//*[@id="wrap"]/div[2]/div[2]/div[2]/div[1]/table/tr[2]/td[2]/a/text()').get()
        series = response.xpath('//*[@id="wrap"]/div[2]/div[2]/div[2]/div[1]/table/tr[3]/td[2]/a/text()').get()
        launch_date = response.xpath('//*[@id="wrap"]/div[2]/div[2]/div[2]/div[1]/table/tr[7]/td[2]/text()').get()
        racket_weight = response.xpath(
            '//*[@id="wrap"]/div[2]/div[2]/div[2]/div[2]/table/tr[3]/td[2]/text()').get()
        racket_length = response.xpath(
            '//*[@id="wrap"]/div[2]/div[2]/div[2]/div[2]/table/tr[4]/td[2]/text()').get()
        grip_thickness = response.xpath(
            '//*[@id="wrap"]/div[2]/div[2]/div[2]/div[2]/table/tr[5]/td[2]/text()').get()
        stringing_tension = response.xpath(
            '//*[@id="wrap"]/div[2]/div[2]/div[2]/div[2]/table/tr[7]/td[2]/text()').get()
        item_data = RacketItem(
            eid=eid,
            name_usuall=name_usuall,
            name_official=name_official,
            brand=brand,
            series=series,
            launch_date=launch_date,
            racket_weight=racket_weight,
            racket_length=racket_length,
            grip_thickness=grip_thickness,
            stringing_tension=stringing_tension
        )
        yield scrapy.Request(
            url=f"https://www.badmintoncn.com/cbo_eq/view_buy.php?eid={eid}",
            callback=self.parse_price,
            headers=self.headers,
            meta={'item': item_data}
        )

    def parse_price(self, response):
        item = response.meta['item']
        price_new = response.xpath('//*[@id="wrap"]/div[2]/div[2]/div[2]/table/tr/td[1]/strong/text()').get().strip()
        price_old = response.xpath('//*[@id="wrap"]/div[2]/div[2]/div[2]/table/tr/td[2]/strong/text()').get().strip()
        # 确保价格信息存在并去除空格
        if price_new:
            item['price_new'] = price_new.strip()
        else:
            item['price_new'] = None  # 如果价格为空，可以设置为 None 或默认值

        if price_old:
            item['price_old'] = price_old.strip()
        else:
            item['price_old'] = None  # 如果价格为空，可以设置为 None 或默认值

        # 返回完整的 item
        yield item