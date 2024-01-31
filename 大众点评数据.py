#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import re
from lxml import etree
import time
import random


# 请求头信息
headers = {
    "Host": "www.dianping.com",
    "If-Modified-Since": "Mon, 18 Mar 2019 09:08:05 GMT",
    "If-None-Match": '"12d4baaa94fa976135ecc38c0f4ff2a6"',
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3100.0 Safari/537.36"
}





# In[2]:


# 获取评论数
def get_score(response):
    score_str = re.findall('<span id="reviewCount" class="item">(.*?)</span>', response)[0]
    score_str=score_str.replace('/d>','').replace('<d class="num"','').replace('条评论','')
    score_str=get_digital(score_str)
    
    return score_str



# In[3]:


def get_num(response):

    score_str = re.findall('<span class="item">(.*?)</span>', response)
    # 获取口味评分
    taste=score_str[0].replace('<d class="num"','').replace('/d>','').replace('口味: ','')
    taste=get_digital(taste)
    # 获取环境评分
    envir=score_str[1].replace('<d class="num"','').replace('/d>','').replace('环境: ','')
    envir=get_digital(envir)
    # 获取服务评分
    serv=score_str[2].replace('<d class="num"','').replace('/d>','').replace('服务: ','')
    serv=get_digital(serv)
    #获取营业时间
    service_time=score_str[3].replace('<svgmtsi class="shopdesc"','').replace('/svgmtsi>','').replace('<svgmtsi class="hours"','').replace('至','')
    service_time=get_digital2(service_time)
    
    return taste,envir,serv,service_time


    
    

    


    


# In[4]:


#获取人均价格
def get_Price(response):
    price_str = re.findall('<span id="avgPriceTitle" class="item">(.+?)</span>', response)[0]
    
    price_str=price_str.replace('/d>','').replace('<d class="num"','').replace('人均: ','').replace('元','')
    price_str=get_digital(price_str)

    
    return price_str


# In[5]:


def get_star(response):
    star= re.findall('<span title="(.*?)" class="mid-rank-stars mid-str',response)[0]

    return star


# In[6]:


def get_branch(response):
    branch=re.findall('查看全部(.*?)家分店',response)[0]
    branch=branch[-2:]
    branch=branch.replace('部','')

    return branch


# In[7]:


def get_address(response):
    address=re.findall('address: "(.*?)",',response)[0]

    return address


# In[8]:


def get_region(response):
    region=re.findall('mainRegionId:(.*?),',response)[0]

    return region


# In[9]:


def get_name(response):
    if re.findall('shopName: "(.*?)"',response)[0] !=[]:
        name=re.findall('shopName: "(.*?)"',response)[0]
        return name
    else:
        return null


# In[10]:


# 数字解密
def get_digital(id):
    id=id.replace('>&#xe47f;<','5')
    id=id.replace('>&#xed64;<','3')
    id=id.replace('>&#xe8f0;<','4')
    id=id.replace('>&#xe587;<','7')
    id=id.replace('>&#xe6fc;<','0')
    id=id.replace('>&#xf1f9;<','6')
    id=id.replace('>&#xecf4;<','2')
    id=id.replace('>&#xed5f;<','9')
    id=id.replace('>&#xe07a;<','8')
    return id


# In[11]:


# 时间数字解密
def get_digital2(id):
    id=id.replace('>&#xe47f;<','5')
    id=id.replace('>&#xf717;<','3')
    id=id.replace('>&#xf56b;<','4')
    id=id.replace('>&#xf043;<','7')
    id=id.replace('>&#xef11;<','0')
    id=id.replace('>&#xea4f;<','6')
    id=id.replace('>&#xf357;<','2')
    id=id.replace('>&#xed5f;<','9')
    id=id.replace('>&#xe07a;<','8')
    return id


# In[12]:


import csv
def run(url):
    with open('data.csv', 'a', newline='',encoding='utf-8-sig') as csvFile:
        writer = csv.writer(csvFile)
        #writer.writerow(['店铺名称','地址','区域号','分店数','人均价格','评论数','口味','环境','服务','营业时间','星级'])
        for u in url:
            new=list()
            response = requests.get(u, headers=headers).text
            new.append(get_name(response))
            new.append(get_address(response))
            new.append(get_region(response))
            new.append(get_branch(response))
            new.append(get_Price(response))
            new.append(get_score(response))
            new.append(get_num(response)[0])
            new.append(get_num(response)[1])
            new.append(get_num(response)[2])
            new.append(get_num(response)[3])
            new.append(get_star(response))
            print(new)
            writer.writerow(new)
            time.sleep(random.randint(30,40))
            
            
        
        


# In[17]:


from bs4 import BeautifulSoup as bs
from urllib.request import urlopen
html=bs(open('D:/大学courses/网页/g110.html',encoding='utf-8'),features='html.parser') 


# In[18]:


web=list()
for element in html.find_all('a',{'data-click-name':'shop_title_click'}):
    web.append(element.get('href'))


# In[19]:


for n in range(2,51):
    html=bs(open('D:/大学courses/网页/g110p%s.html'%str(n),encoding='utf-8'),features='html.parser') 
    for element in html.find_all('a',{'data-click-name':'shop_title_click'}):
        web.append(element.get('href'))


# In[22]:


run(web)

