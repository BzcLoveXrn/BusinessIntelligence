# Define here the models for your spider middleware
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/spider-middleware.html
import random

import requests
from scrapy import signals

# useful for handling different item types with a single interface
from itemadapter import is_item, ItemAdapter
from twisted.internet import task
import random
from scrapy.downloadermiddlewares.retry import RetryMiddleware
from scrapy.utils.response import response_status_message


class BadmintoncnSpiderMiddleware:
    # Not all methods need to be defined. If a method is not defined,
    # scrapy acts as if the spider middleware does not modify the
    # passed objects.

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_spider_input(self, response, spider):
        # Called for each response that goes through the spider
        # middleware and into the spider.

        # Should return None or raise an exception.
        return None

    def process_spider_output(self, response, result, spider):
        # Called with the results returned from the Spider, after
        # it has processed the response.

        # Must return an iterable of Request, or item objects.
        for i in result:
            yield i

    def process_spider_exception(self, response, exception, spider):
        # Called when a spider or process_spider_input() method
        # (from other spider middleware) raises an exception.

        # Should return either None or an iterable of Request or item objects.
        pass

    def process_start_requests(self, start_requests, spider):
        # Called with the start requests of the spider, and works
        # similarly to the process_spider_output() method, except
        # that it doesn’t have a response associated.

        # Must return only requests (not items).
        for r in start_requests:
            yield r

    def spider_opened(self, spider):
        spider.logger.info("Spider opened: %s" % spider.name)


class BadmintoncnDownloaderMiddleware:
    # Not all methods need to be defined. If a method is not defined,
    # scrapy acts as if the downloader middleware does not modify the
    # passed objects.

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_request(self, request, spider):
        # Called for each request that goes through the downloader
        # middleware.

        # Must either:
        # - return None: continue processing this request
        # - or return a Response object
        # - or return a Request object
        # - or raise IgnoreRequest: process_exception() methods of
        #   installed downloader middleware will be called
        return None

    def process_response(self, request, response, spider):
        # Called with the response returned from the downloader.

        # Must either;
        # - return a Response object
        # - return a Request object
        # - or raise IgnoreRequest
        return response

    def process_exception(self, request, exception, spider):
        # Called when a download handler or a process_request()
        # (from other downloader middleware) raises an exception.

        # Must either:
        # - return None: continue processing this exception
        # - return a Response object: stops process_exception() chain
        # - return a Request object: stops process_exception() chain
        pass

    def spider_opened(self, spider):
        spider.logger.info("Spider opened: %s" % spider.name)


PROXY_POOL = []


class RandomProxyMiddleware:
    def __init__(self):
        # 在这里定义你的代理池

        self.proxy_api_url = "http://route.xiongmaodaili.com/xiongmao-web/api/glip?secret=633668262f24a9aea1c33e83c9a4af73&orderNo=GL20241221153130032fpNrJ&count=6&isTxt=0&proxyType=1&returnAccount=1"
        self.loop = task.LoopingCall(self.update_proxies)
        self.loop.start(290)  # 每 300 秒（即 5 分钟）调用一次 update_proxies

    def process_request(self, request, spider):
        # 随机选择一个代理
        proxy = random.choice(PROXY_POOL)
        request.meta['proxy'] = proxy
        spider.logger.debug(f"Using proxy {proxy} for {request.url}")

    def update_proxies(self):
        # 更新代理池
        response = requests.get(self.proxy_api_url)
        if response.status_code == 200:
            json_data = response.json()
            proxys = json_data['obj']
            PROXY_POOL.clear()
            print("代理更新了", proxys)
            for proxy in proxys:
                PROXY_POOL.append(f"https://xmdl743c:xmdl93fa@{proxy['ip']}:{proxy['port']}")
        else:
            print("代理更新失败", response.text)


class CustomRetryMiddleware(RetryMiddleware):
    def __init__(self, settings):
        super().__init__(settings)

    def _retry(self, request, reason, spider):
        retry_times = request.meta.get('retry_times', 0) + 1
        if retry_times <= self.max_retry_times:
            request.meta['retry_times'] = retry_times

            # 如果是403错误，尝试更换代理
            if '403' in reason:
                new_proxy = self.get_new_proxy()
                request.meta['proxy'] = new_proxy
                spider.logger.info(f"Switching proxy to {new_proxy} for {request.url}")

            return self._retry_request(request, spider)
        else:
            spider.logger.warning(f"Max retries reached for {request.url}, giving up.")
            return None

    def get_new_proxy(self):
        """从代理池获取一个新的代理"""
        return random.choice(PROXY_POOL)

    def process_response(self, request, response, spider):
        if response.status in self.retry_http_codes:
            reason = response_status_message(response.status)
            return self._retry(request, reason, spider)
        return response

    def process_exception(self, request, exception, spider):
        if isinstance(exception, (TimeoutError, ConnectionError)):
            reason = str(exception)
            return self._retry(request, reason, spider)
        return None
