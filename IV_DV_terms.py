__author__ = 'zhangye'
#construct a IV/DV terms dictionary
import  os
import xlrd
from textblob import TextBlob
dir = "IV_DV"
i = 0
for file_name in os.listdir("1. excel files"):
    if file_name.endswith(".xls"):
            IV_DV_file = open("IV_DV/"+file_name[:-4]+".txt",'wb')
            book = xlrd.open_workbook("1. excel files/"+file_name)
            first_sheet = book.sheet_by_index(0)
            terms = []

            DV_title = first_sheet.cell(14,5).value
            if(DV_title!=-9 and DV_title!=0):
                DV_title = DV_title.encode('ascii','ignore')
                terms += list(TextBlob(DV_title.lower()).correct().words)

            IV_title = first_sheet.cell(15,5).value
            if(IV_title!=-9 and IV_title!=0):
                IV_title = IV_title.encode('ascii','ignore')
                terms += [w for w in list(TextBlob(IV_title.lower()).correct().words)]

            DV = first_sheet.cell(39,5).value
            if(DV!=-9 and DV!=0):
                DV = DV.encode('ascii','ignore')
                terms += [w for w in list(TextBlob(DV.lower()).correct().words)]

            IV = first_sheet.cell(19,5).value
            if(IV!=-9 and IV!=0):
                IV = IV.encode('ascii','ignore')
                terms += ([w for w in list(TextBlob(IV.lower()).correct().words)])

            IV_DV_file.write("\t".join(terms)+"\n")
            IV_DV_file.close()
