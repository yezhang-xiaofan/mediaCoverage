__author__ = 'zhangye'
from bs4 import BeautifulSoup
import requests
#this program parses press release for Oxford research article journal
root = "http://www.ox.ac.uk"
root_url = "http://www.ox.ac.uk/news-and-events/for-journalists"
response = requests.get(root_url)
soup = BeautifulSoup(response.text)
links = soup.find_all("div",class_="field-group-format group_news_listing_details field-group-div group-news-listing-details speed-none effect-none")
links1 = [a.find('a') for a in links]
links2 = [root+a.attrs.get('href') for a in links1]

for i in range(1,7):
    response = requests.get(root_url+"?page="+str(i))
    soup = BeautifulSoup(response.text)
    links = soup.find_all("div",class_="field-group-format group_news_listing_details field-group-div group-news-listing-details speed-none effect-none")
    links1 = [a.find('a') for a in links]
    links2 = links2 + [root+a.attrs.get('href') for a in links1]

dir = "PressRelease_Oxford"
i = 1
for link in links2:
    soup = BeautifulSoup(requests.get(link).text)
    category = soup.find("div",class_="tags")
    if(category==None):
        continue
    category = category.get_text()
    print category
    if("Health" not in category and  "health" not in category):
        continue
    title = soup.find("title")
    sentences = soup.find_all("span",class_="field-item-single")
    res_sentences = []
    file = open(dir+"/"+str(i)+".txt",'w')
    file.write(title.get_text().encode("ascii","ignore")+"\n")
    for s in sentences:
        if s==[]:
            continue
        sen = s.find_all(["p","li"])
        for s in sen:
            #print(s.get_text().encode("ascii","ignore"))
            file.write(s.get_text().encode("ascii","ignore")+"\n")
    i += 1
    file.close()


