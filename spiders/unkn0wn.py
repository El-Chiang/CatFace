import requests
import json


def download(url,name):
    img = requests.get(url)
    file_name = 'images/'+name+'.jpg'
    fp = open(file_name,'wb')
    fp.write(img.content)

global id
id = 1

def getimg(page):

    global id

    uid = '1774676624'
    containerid = '1076031774676624'
    url = 'https://m.weibo.cn/api/container/getIndex?type=uid&value='+uid+'&containerid='+containerid+'&page='+str(page)
    # print (url)
    r = requests.get(url)
    param = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
        'Cache-Control': 'max-age=0',
        'Connection': 'keep-alive',
        'Cookie': '_T_WM=e68184ccfdccfabba60bc7fb18685780; TMPTOKEN=u96VPsWXJC1hhYFyKrCgq9ksFuaWasIgQRBUNjVn6gDgdGu5RglDH09oEEzEwfGT; SUB=_2A2536PaBDeRhGeVK6FYV9y7EyDyIHXVVEprJrDV6PUJbkdAKLUnRkW1NTEG31oVaAJarfAnpgLckdvP2ZN7Tc2mK; SUHB=0QSJ2a2wj4E2cZ; SCF=AqQT15d3QWteQbgFFpUiz4zCML5axc7RmZ_MgmAs6fCgs9KJOZ4zKSD4N0_m2xwL2EPzJ-895t4QJ2jXT4zSGks.; SSOLoginState=1525450449; M_WEIBOCN_PARAMS=featurecode%3D20000320%26oid%3D4236107610886685%26luicode%3D10000011%26lfid%3D1076031774676624%26fid%3D1076031774676624%26uicode%3D10000011; MLOGIN=1; WEIBOCN_FROM=1110006030',
        'Host': 'm.weibo.cn',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:59.0) Gecko/20100101 Firefox/59.0',
    }
    data = json.loads(r.text)

    for card in data['data']['cards']:
        if 'mblog' in card:
            if 'pics' in card['mblog']:
                for pic in card['mblog']['pics']:
                    download(pic['url'],str(id))
                    id = id + 1
            elif 'retweeted_status' in card['mblog']:
                text = card['mblog']['retweeted_status']['text']
                if '皮皮' or '萌萌' in text:
                    if 'pics' in card['mblog']['retweeted_status']:
                        for pic in card['mblog']['retweeted_status']['pics']:
                            download(pic['url'],str(id))
                            id = id + 1

for i in range(1,20):
    getimg(i)