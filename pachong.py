import asyncio
import re

import aiohttp
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options


def ks_download_uel(image_urls):
    async def download_images(url_list):
        async with aiohttp.ClientSession() as session:
            global k
            for url in url_list:
                try:
                    async with session.get("https:" + url) as response:  # "https:" + url 进行网址拼接
                        response.raise_for_status()
                        file_path = fr"F:\project\猫12分类\data\孟买猫\{k}.jpg"  # 指定保存地址
                        with open(file_path, 'wb') as file:
                            while True:
                                chunk = await response.content.read(8192)
                                if not chunk:
                                    break
                                file.write(chunk)
                    print(f"已经完成 {k} 张")
                except Exception as e:
                    print(f"下载第 {k} 张出现错误 ：{str(e)}")
                k += 1  # 为下一张做标记

    # 创建事件循环对象
    loop = asyncio.get_event_loop()
    # 调用异步函数
    loop.run_until_complete(download_images(image_urls))


if __name__ == '__main__':
    base_url = 'https://www.vcg.com/creative-image/mengmaimao/?page={page}'  # "buoumao"为布偶猫的拼音，如果想搜索其他品种的猫，直接更改拼音就可以
    edge_options = Options()
    edge_options.add_argument("--headless")  # 不显示浏览器敞口， 加快爬取速度。
    edge_options.add_argument("--no-sandbox")  # 防止启动失败
    driver = webdriver.Edge(options=edge_options)

    k = 1  # 为保存的每一种图片做标记
    for page in range(1, 5):  # 每一页150张，十页就够了。
        if page == 1:  # 目的是就打开一个网特，减少内存开销
            driver.get(base_url.format(page=page))  # 开始访问第page页
        elements = driver.find_elements(By.XPATH,
                                        '//*[@id="imageContent"]/section[1]')  # 将返回 //*[@id="imageContent"]/section/div 下的所有子标签元素
        urls_ls = []  # 所要的图片下载地址。
        for element in elements:
            html_source = element.get_attribute('outerHTML')
            urls_ls = re.findall('data-src="(.*?)"', str(html_source))  # 这里用了正则匹配，可以加快执行速度

        #  下面给大家推荐一个异步快速下载图片的方法， 建议这时候尽量多提供一下cpu和内存为程序
        ks_download_uel(urls_ls)

        driver.execute_script(f"window.open('{base_url.format(page=page)}', '_self')")  # 在当前窗口打开新网页，减少内存使用
    driver.quit()  # 在所有网页访问完成后退出 WebDriver