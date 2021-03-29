import scrapy
import os

# Scrape data using scrapy. Outputs results to filepath specified by class attribute dest_file

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
