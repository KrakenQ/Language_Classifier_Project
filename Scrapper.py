import requests
import os
from pywikiapi import wikipedia
list_of_languages = ['ady', 'an', 'ast', 'av', 'az', 'ba', 'bar', 'be', 'bg', 'br', 'bs', 'ca', 'ce', 'co', 'crh', 'cs',
                     'csb', 'cv', 'cy', 'da', 'de', 'el', 'en', 'es', 'et', 'eu', 'ext', 'fi', 'fo', 'fr', 'frp', 'fur',
                     'ga', 'gag', 'gd', 'gl', 'hr', 'hu', 'inh', 'is', 'it', 'kbd', 'kk', 'krc', 'ksh', 'kv', 'la',
                     'lad', 'lb', 'li', 'lij', 'lmo', 'lt', 'lv', 'mdf', 'mk', 'mt', 'mwl', 'myv', 'nap', 'nds', 'nl',
                     'no', 'oc', 'os', 'pcd', 'pl', 'pt', 'ro', 'ru', 'sc', 'scn', 'sco', 'se',
                     'sk', 'sl', 'sq', 'sr', 'sv', 'szl', 'tr', 'tt', 'udm', 'uk', 'vec', 'wa', 'xal', 'yi']
def scrap():
    path = "Dataset"
    try:
        for i in list_of_languages:
            os.makedirs('Dataset/' + i)
    except OSError:
        print("Creation of the directories %s failed" % path)
    else:
        print("Successfully created the directories %s" % path)

    for i in list_of_languages:
        print('Robię język ' + i)
        site = wikipedia(i)
        for j in range(100):
            for r in site.query(generator='random', grnnamespace=0, grnlimit = 1, prop=['extracts'], explaintext = 1):
                for page in r.pages:
                    title = page.title
                    title = "".join(x for x in title if x.isalnum())
                    file = open('Dataset/' + i + '/' + title + '.txt', "w", encoding="utf-8")
                    file.write(page.extract)
                break
