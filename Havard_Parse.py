__author__ = 'zhangye'
#this program parses Harvard press release from the website
from bs4 import BeautifulSoup
import requests
root = "http://www.hsph.harvard.edu/news/press-release/"
year = [2008,2009,2010,2011,2012,2013,2014,2015]
links2 = []

def extract_links(soup):
    link = soup.find_all("h2",class_="entry-title")
    links1 = [a.find('a') for a in link]
    return [a.attrs.get('href') for a in links1]

for i in range(len(year)):
    page = root+str(year[i])
    response = requests.get(page)
    soup = BeautifulSoup(response.text)
    page_numbers = soup.find_all("a",class_="page-numbers")
    links2 += extract_links(soup)
    if(page_numbers>1):
        for j in range(1,len(page_numbers)):
            new_page = page+"/page/"+page_numbers[j].get_text().encode('ascii','ignore')
            response = requests.get(new_page)
            new_soup = BeautifulSoup(response.text)
            links2 += extract_links(new_soup)

PR_Harvard = []

#if the parent tag is div, the sentences are not in the main body
def parent_div(tag):
    if (tag.name=='p' and not tag.has_attr('class')):
        if not (tag.parent.has_attr('class') and tag.parent['class'][0]=='textwidget'):
            return True
    elif (tag.name=='ul' and not tag.has_attr('class')):
        return True
    else:
        return False


i = 1
dir = "PR_Harvard"
for link in links2:
    soup = BeautifulSoup(requests.get(link).text)
    allsentences = soup.find_all(parent_div)
    file = open(dir+"/"+str(i)+".txt",'w')
    title = soup.find('title')
    file.write(title.get_text().encode("ascii","ignore").split('|')[0]+"\n")
    for s in allsentences:
        if(s.get_text()=='###'):break
        file.write(s.get_text().encode("ascii","ignore")+"\n")
    i += 1
    file.close()

