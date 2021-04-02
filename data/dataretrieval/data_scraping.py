import scrapy
from scrapy import cmdline
import os
import requests
from zipfile import ZipFile

# WARNING: URLs and filenames will likely change as turbine database is
# updated, highly advised to use the already prepared clean dataset.
# See README for link to the prepared, cleaned data.

url = 'https://eerscmap.usgs.gov/uswtdb/assets/data/uswtdbCSV.zip'
dest_file = 'data/rawdata/windturbines.zip'

if os.path.exists(dest_file):
    os.remove(dest_file)

r = requests.get(url)
with open(dest_file,'wb') as code:
    code.write(r.content)

with ZipFile(dest_file,'r') as zip:
    zip.extractall(path='data/rawdata')

os.remove(dest_file)
os.remove('data/rawdata/CHANGELOG.txt')
os.rename('data/rawdata/uswtdb_v3_3_20210114.csv','data/rawdata/WindRawRetrieve.csv')


# WARNING: URLs and filenames will likely change as solar farm database is
# updated, highly advised to use the already prepared clean dataset.
# See README for link to the prepared, cleaned data.

class SolarSpider(scrapy.Spider):
    """Scrapes solarprojects.anl.gov for list of all large solar projects and saves file to
    path specified by dest_file class attribute"""

    name = "solar_spider"
    dest_file = 'data/rawdata/SolarRawRetrieve.csv'

    if os.path.exists(dest_file):
        os.remove(dest_file)

    def start_requests(self):
        url = 'https://solarprojects.anl.gov/details.cfm?id='
        urls = (url + str(val) for val in range(1000))
        for val in urls:
            yield scrapy.Request(url = val, callback = self.parse )

    def parse(self, response):
        data_values = response.xpath('//td/text()').extract()
        if not data_values:
            pass
        else:
            with open(self.dest_file, 'a') as f:
                f.writelines([data + '\t' for data in data_values])
                f.writelines('\n')


cmdline.execute('scrapy runspider data/dataretrieval/data_scraping.py'.split())