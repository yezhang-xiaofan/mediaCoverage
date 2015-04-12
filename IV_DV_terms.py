__author__ = 'zhangye'
#construct a IV/DV terms dictionary
import  os
import xlrd
from textblob import TextBlob
IV_DV_file = open("IV_DV.txt",'wb')
for file_name in os.listdir("1. excel files"):
    if file_name.endswith(".xls"):
        book = xlrd.open_workbook("1. excel files/"+file_name)
        first_sheet = book.sheet_by_index(0)
        if file_name.endswith(".xls"):
            DV = first_sheet.cell(39,5).value
            IV = first_sheet.cell(19,5).value
            if(DV==-9 or IV==-9):
                continue
            DV = DV.encode('ascii','ignore')
            IV = IV.encode('ascii','ignore')
            DV = " ".join([w.lower() for w in list(TextBlob(DV).correct().words)])
            IV = " ".join([w.lower() for w in list(TextBlob(IV).correct().words)])
            IV_DV_file.write(DV+"\n")
            IV_DV_file.write(IV+"\n")
IV_DV_file.close()
