'''
Code to process the Chambers et al. dataset.
'''
import random
import csv 
import pdb
import os 
import operator 
import xlrd
import re 

import Bio
from Bio import Entrez
from Bio import Medline
import pubmedpy


def xls_files_to_csvs(excel_dir="1. excel files"):
    '''
    convert excel files provided by Chambers, which contain the
    labels, to CSVs.
    '''
    for xls_f in [f for f in os.listdir(excel_dir) if f.endswith(".xls")]:
        xls_file_to_csv(os.path.join(excel_dir, xls_f))


def xls_file_to_csv(excel_path):
    wb = xlrd.open_workbook(excel_path)
    sh = wb.sheet_by_name('Sheet1')
    with open(excel_path.replace(".xls", ".csv"), 'wb') as out_f:
        wr = csv.writer(out_f, quoting=csv.QUOTE_ALL)
        for row in xrange(sh.nrows):
            this_row = []
            for col in xrange(sh.ncols):
                val = sh.cell_value(row,col)
                ### grrr utf8
                if isinstance(val, str) or isinstance(val, unicode):
                    this_row.append(val.encode('utf8'))
                else:
                    this_row.append(val)
            wr.writerow(this_row)


def generate_all_article_info(csv_dir="1. excel files"):
    '''
    1. Parse all .csv files in the directory specified by csv_dir, and, 
    2. Generate corresponding article information files that include 
        PMIDs and titles (citation info) for articles annotated in csv 
        files. 
    '''
    failures = []
    for csv_f in [f for f in os.listdir(csv_dir) if f.endswith(".xls")]:
        cur_title = get_article_title(os.path.join(csv_dir, csv_f))
        #get pubmid from title
        pmid = pubmedpy.get_pmid_from_title(cur_title)
        if pmid: 
            print "matched %s for %s." % (pmid, csv_f)
            citation_info = list(pubmedpy.fetch_articles([pmid]))[0]
            title, mesh, authors, abstract, affiliation, journal, volume, article = get_fields_from_article(citation_info)
            with open(os.path.join(csv_dir,csv_f).replace(".xls", ".article_info.txt"), 'wb') as out_f:
                csv_writer = csv.writer(out_f)
                csv_writer.writerow([pmid, title, mesh, authors, abstract, affiliation, journal, volume])
        else:
            #print "!!! failure !!! could not find PubMed entry for %s" % cur_title
            failures.append(cur_title)

    print "ok! %s failures (see below, if any). \n %s" % (len(failures), "\n".join(failures))


def get_fields_from_article(article):    
    title, mesh, authors, abstract, affiliation, journal, volume, issue = ["Missing"]*8
    if "PMID" not in article:
        print "error -- couldn't get this article..."
        #pdb.set_trace()
        #return title, mesh, authors, abstract, affiliation
        return False

    print "\n --" + article["PMID"] + "-- \n"
    if "TI" in article:
        title = article["TI"]

    if "MH" in article:
        # mesh terms sometimes contain spaces,
        # so should split on something else!
        #mesh = u" ".join(article["MH"])
        mesh = u"\n".join(article["MH"])

    if "AB" in article:
        abstract = article["AB"]

    if "AU" in article:
        authors = u" and ".join(article["AU"])

    # get journal and volume!!!
    if "JT" in article:
        journal = article["JT"]

    if "VI" in article:
        volume = article["VI"]

    if "IP" in article:
        issue = article["IP"]


    ###
    # try to get the 'institution'. a note about this:
    # the "AD" field has this info, but it's largely
    # unstructured. e.g.:
    # 
    # Department of Health Behavior, School of Public Health, University of Alabama at Birmingham, 1665 University Blvd, 227 RPHB, Birmingham, AL 35294, USA. rlanzi@uab.edu  
    # 
    # instead of trying to grab the uni off of this, we look for 
    # an email (which seems to usually be available) and then
    # for a .edu or .org
    ####
    
    if "AD" in article:
        affiliation_str = article["AD"]
        match = re.search(r'[\w\.-]+@[\w\.-]+', affiliation_str)
        
        # get the email
        if match is not None:    
            email = match.group(0)
            affiliation = email.split("@")[1]
            domains = [".edu", ".org", ".nl", ".gov"]
            
            for d in domains:
                if d in affiliation:
                    affiliation = affiliation.split(d)[0].split(".")[-1]
                    print "found institution %s for %s" % (affiliation, affiliation_str)
                    break
            else:
                affiliation = "Missing"

        else:
            print "couldn't find an email in this affiliation str..."
            print affiliation_str

    return title, mesh, authors, abstract, affiliation, journal, volume, article

def get_article_title(csv_path):
    '''
    identify and return the original article title 
    matching the press release and annotations found 
    in the file at csv_path.
    '''
    book = xlrd.open_workbook(csv_path)
    first_sheet = book.sheet_by_index(0)
    return first_sheet.cell(1,1).value



'''
matched sampling stuff
'''
def perform_matched_sampling(matched_article_dir="7. Matched samples"):
    article_files = [f for f in os.listdir(".") if f.endswith("article_info.txt")]

    # now get info for each of the pmids and 
    # sample articles accordingly
    for article_file in article_files:
        print "\n\nperforming matched sampling for article: %s...\n" % article_file
        pmid = _get_pmid_from_article(article_file)
        journal, volume, issue = get_info_for_pmid(pmid)

        # now match based on these.
        matched_articles = matched_sample(journal, volume, issue, pmid)

        # ok -- now write them out!
        # dump the samples to disk
        for i, matched_article in enumerate(matched_articles):
            out_path = os.path.join(matched_article_dir, 
                            article_file + "_matched_article_%s.csv" % i)
            with open(out_path, 'wb') as outf:
                title, mesh, authors, abstract, affiliation, journal, volume, article = get_fields_from_article(matched_article)
                csv_writer = csv.writer(outf)
                csv_writer.writerow([matched_article["PMID"], title,
                            journal, authors, affiliation, abstract, mesh])



def get_article_from_pmid(pmid):
    return list(pubmedpy.fetch_articles([str(pmid)]))[0]

def get_info_for_pmid(pmid):
    # first get the article
    article = get_article_from_pmid(pmid)

    # now parse out fields of interst.
    return get_info_for_article(article)

def _get_pmid_from_article(f):
    PMID_COLUMN = 0 
    with open(f, 'rU') as article_info_f:
        reader = csv.reader(article_info_f)
        return reader.next()[PMID_COLUMN]

def get_info_for_article(article):    
    '''
    This is information to be used for the matched 
    sampling
    '''
    journal, volume, issue = [None]*3

    if "JT" in article: 
        journal = article["JT"]

    if "VI" in article:
        volume = article["VI"]

    if "IP" in article:
        issue = article["IP"]

    return journal, volume, issue 


def matched_sample(journal, volume, issue, pmid, k=20):
    # note that we need the PMID to avoid 'sampling' the 
    # article that we're doing the matched sampling for...
    ### 
    # !!! note we want to try this matching on the issue as well here!
    ###
    handle = Entrez.esearch(db="pubmed", retmax=200, term='''%s [journal] %s[volume] %s[issue]''' % (
                                journal, volume, issue))

    # Arch Pediatr Adolesc Med [journal] 2012[year] 
    record = Entrez.read(handle)
    print "found %s articles for %s, %s (%s)" % (record["Count"], journal, volume, issue)
    ids = record["IdList"]
    random.shuffle(ids)

    # now we do a filtering step
    def _check_pages(a, MIN_PAGES=5):
        # PG is the pages field.
        if "PG" in a:
            pages = a["PG"]
            if "-" in pages:
                # like 993-1001
                try:
                    start, end = pages.split("-")
                    start = start.strip()
                    end = end.strip()

                    #if len(start) != len(end):
                    start_int = int(start)
                    end_int = int(end)
                    start_digits, end_digits = len(start), len(end)
                    if start_digits <= end_digits:
                        return (end_int - start_int) > MIN_PAGES

                    # otherwise, assume that we need to prepend
                    # the most sig digits of start to the end
                    # eg., 493-4    
                    diff = start_digits - end_digits
                    end_adjusted = start[:diff] + end 
                    end_int_adjusted = int(end_adjusted)

                    #pdb.set_trace()
                    return (end_int_adjusted - start_int) > MIN_PAGES
                except:
                    return False
        return False 


    tests = [
        # the stuff that's not indexed by medline seems 
        # to stuff we don't want
        lambda a: a["STAT"] != "PubMed-not-MEDLINE", 
        # don't sample the target article
        lambda a: a["PMID"] != str(pmid),
        # get rid of commentaries
        lambda a: not "CON" in a,
        # page filter: kind of hacky, but we don't want
        # most things that don't span multiple pages.
        # otherwise you get things like brief comments
        # and photos of the month.
        # the 4 here is (an arbitrary) lower bound on the 
        # page count
        lambda a: _check_pages(a)
    ]

    matched_records = []
    for id_ in ids:
        if len(matched_records) >= k:
            return matched_records
        article = get_article_from_pmid(id_)
     
        if all([test(article) for test in tests]):
            print "ok! matched article: %s" % id_ 
            matched_records.append(article)

    return matched_records


'''
def get_fields_from_article(article):    
    title, mesh, authors, abstract, affiliation = ["Missing"]*5
    if "PMID" not in article:
        print "error -- couldn't get this article..."
        #pdb.set_trace()
        #return title, mesh, authors, abstract, affiliation
        return False

    print "\n --" + article["PMID"] + "-- \n"
    if "TI" in article:
        title = article["TI"]

    if "MH" in article:
        mesh = u" ".join(article["MH"])

    if "AB" in article:
        abstract = article["AB"]

    if "AU" in article:
        authors = u" and ".join(article["AU"])

    ###
    # try to get the 'institution'. a note about this:
    # the "AD" field has this info, but it's largely
    # unstructured. e.g.:
    # 
    # Department of Health Behavior, School of Public Health, University of Alabama at Birmingham, 1665 University Blvd, 227 RPHB, Birmingham, AL 35294, USA. rlanzi@uab.edu  
    # 
    # instead of trying to grab the uni off of this, we look for 
    # an email (which seems to usually be available) and then
    # for a .edu or .org
    ####
    
    if "AD" in article:
        affiliation_str = article["AD"]
        match = re.search(r'[\w\.-]+@[\w\.-]+', affiliation_str)
        
        # get the email
        if match is not None:    
            email = match.group(0)
            affiliation = email.split("@")[1]
            domains = [".edu", ".org", ".nl", ".gov"]
            #pdb.set_trace()
            for d in domains:
                if d in affiliation:
                    affiliation = affiliation.split(d)[0].split(".")[-1]
                    print "found institution %s for %s" % (affiliation, affiliation_str)
                    break
            else:
                affiliation = "Missing"

        else:
            print "couldn't find an email in this affiliation str..."
            print affiliation_str

    return title, mesh, authors, abstract, affiliation
'''

def main():
    generate_all_article_info()
if __name__ == '__main__':
    main()