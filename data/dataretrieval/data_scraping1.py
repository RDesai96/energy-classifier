import os
from zipfile import ZipFile
import requests
import scrapy
from scrapy import cmdline


# WARNING: URLs may change as turbine database is updated.
# Changing default filepaths/filenames of data scrapes will cause
# conflicts in the other data/dataretrieval python files.
# Highly advised to use the already prepared clean dataset.
# See README for link to the prepared, cleaned data.

# Retrieving wind turbine data
def get_zip_csv(url,
                file_path,
                csv_name):
    """
    Retrieves zipfile from url and extracts all csv files.
    Saves the extracted csv files at 'file_path + csv_name + int'
    """
    file_zip='data/rawdata/temp_file.zip'
    i=1
    if os.path.exists(file_zip):
        os.remove(file_zip)
    r = requests.get(url)
    with open(file_zip,'wb') as code:
        code.write(r.content)
    with ZipFile(file_zip,'r') as zip:
        zip.extractall(path=file_path)
        for file in zip.filelist:
            if '.csv' in file.filename:
                os.rename(os.path.join(file_path,file.filename),
                          os.path.join(file_path, csv_name + str(i) + '.csv'))
                i += 1
            else:
                os.remove(os.path.join(file_path, file.filename))
    os.remove(file_zip)


get_zip_csv(url='https://eerscmap.usgs.gov/uswtdb/assets/data/uswtdbCSV.zip',
            file_path='data/rawdata',
            csv_name='WindRawRetrieve')


# WARNING: URLs may change as solar farm database is updated.
# Highly advised to use the already prepared clean dataset.
# See README for link to the prepared, cleaned data.

class SolarSpider(scrapy.Spider):
    """
    Scrapes solarprojects.anl.gov for list of all large solar projects and saves file to
    path specified by dest_file class attribute. The scraped website are assumed to have
    the form of of 'self.url_root + integer'.
    """
    name = "solar_spider"
    dest_file = 'data/rawdata/SolarRawRetrieve.csv'
    url_root = 'https://solarprojects.anl.gov/details.cfm?id='

    if os.path.exists(dest_file):
        os.remove(dest_file)

    def start_requests(self):
        url = self.url_root
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


cmdline.execute('scrapy runspider data/dataretrieval/data_scraping1.py'.split())