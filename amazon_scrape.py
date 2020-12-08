from bs4 import BeautifulSoup
from lxml import html
import time
import requests
import csv


def get_title(page_soup):
    try:
        title = page_soup.find("span", attrs={"id": 'productTitle'})
        title_value = title.string
        title_string = title_value.strip()

    except AttributeError:
        title_string = ""

    return title_string


def get_description(page_soup):
    try:
        description_div = page_soup.find("div", attrs={"id": 'productDescription'})
        description_text = description_div.find('p').text.strip()

    except AttributeError:
        description_text = ""

    return description_text


# Function to extract Product Price
def get_price(page_soup):
    try:
        price = page_soup.find("span", attrs={'id': 'priceblock_ourprice'}).string.strip()

    except AttributeError:
        price = ""

    return price


# Function to extract Product Rating
def get_rating(page_soup):
    try:
        rating = page_soup.find("i", attrs={'class': 'a-icon a-icon-star a-star-4-5'}).string.strip()

    except AttributeError:

        try:
            rating = page_soup.find("span", attrs={'class': 'a-icon-alt'}).string.strip()
        except:
            rating = ""

    return rating


# Function to extract Number of User Reviews
def get_review_count(page_soup):
    try:
        review_count = page_soup.find("span", attrs={'id': 'acrCustomerReviewText'}).string.strip()

    except AttributeError:
        review_count = ""

    return review_count


def is_prime(page_soup):
    try:
        priceblock = page_soup.find("tr", attrs={'id': 'priceblock_ourprice_row'})
        ourprice = priceblock.find("span", attrs={'id': 'ourprice_shippingmessage'})
        price_badging = page_soup.find("span", attrs={'id': 'priceBadging_feature_div'})

        print(price_badging)
        prime_icon = price_badging.find("i", {'class': 'a-icon a-icon-prime'})
        if prime_icon:
            return True

    except AttributeError:
        prime_icon = ""
    return prime_icon


def get_availability(page_soup):
    try:
        available = soup.find("div", attrs={'id': 'availability'})
        available = available.find("span").string.strip()

    except AttributeError:
        available = ""

    return available


if __name__ == '__main__':
    HEADERS = ({'User-Agent':
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36 X-Middleton/1'})

    BASE_URL = "https://www.amazon.it/s?k=computer"
    PAGE_URL = "https://www.amazon.it/s?k=computer&page="
    # products_asin = set()
    print("calling: " + BASE_URL)
    first_webpage = requests.get(BASE_URL, headers=HEADERS)
    # first_soup = BeautifulSoup(first_webpage.content,"lxml")
    # last_page = first_soup.find("div",attrs = {"id":"search"})
    # last_page_2 = last_page.find("div",{"class":"a-section a-spacing-none a-padding-base"})
    # last_page_3 = last_page_2.find("li",{"class":"a-disabled"}).text
    first_tree = html.fromstring(first_webpage.content)
    # first_asins = first_tree.xpath('//*[@data-asin]/@data-asin')
    # products_asin.update(first_asins)
    pages = first_tree.xpath('//*[@id="search"]/div[1]/div[2]/div/span[3]/div[2]/div[59]/span/div/div/ul/li[6]/text()')
    print("total pages: \n")
    print(pages)

    # main_soup = BeautifulSoup(first_webpage.content, "lxml")
    # products_elems = main_soup.find_all("div", {
    #     "class": "s-expand-height s-include-content-margin s-border-bottom s-latency-cf-section"})
    #
    # for prod in products_elems:
    #     prod_anchor = prod.find("a", {"class": "a-link-normal a-text-normal"})
    #     prod_name = prod.find("span", {"class": "a-size-base-plus a-color-base a-text-normal"})
    #     is_prime = prod.find("i", attrs={"aria-label": "Amazon Prime"})
    #     if is_prime:
    #         print(prod_anchor.get("href"))
    #
    #     product_url = "amazon.it/"+str(prod_anchor.get("href"))
    #     product_name = prod_name.text.strip()
    #
    #     # print(prod_anchor.get("href"))  # OK
    #     # print(prod_name.text)

    with open("amazon_products.tsv", mode="w", newline='', encoding="utf-8") as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for i in range(1, 8):
            print("PAGE: " + str(i) + "\n")
            time.sleep(20)
            new_page = requests.get(PAGE_URL + str(i), headers=HEADERS)
            print(new_page)
            main_soup = BeautifulSoup(new_page.content, "lxml")
            products_elems = main_soup.find_all("div", {
                "class": "s-expand-height s-include-content-margin s-border-bottom s-latency-cf-section"})

            print(len(products_elems))
            for prod in products_elems:
                soup_anchor = prod.find("a", {"class": "a-link-normal a-text-normal"})
                soup_name = prod.find("span", {"class": "a-size-base-plus a-color-base a-text-normal"})

                is_prime = prod.find("i", attrs={"aria-label": "Amazon Prime"})
                prod_prime = 0
                if is_prime:
                    prod_prime = 1

                product_url = "https://amazon.it/" + str(soup_anchor.get("href"))
                product_name = soup_name.text.strip()
                if product_url == '': print("empty")
                print(product_url)
                time.sleep(5)
                prod_page = requests.get(product_url, headers=HEADERS)
                soup = BeautifulSoup(prod_page.content, "lxml")
                # out_file.write(product_name+"\t"+product_url+"\t"+get_price(soup)+"\t"+get_description(soup)+"\t"+str(prod_prime)+"\n")
                tsv_writer.writerow([product_name, product_url, get_price(soup), get_description(soup), prod_prime])

    # pages = first_tree.xpath('//*[@id="search"]/div[1]/div[2]/div/span[3]/div[2]/div[59]/span/div/div/ul/li[6]/text()')
    # print("total pages: \n")
    # print(int(pages[0]))
    # print("PAGE: 1\n")
    # for i in range(2, int(pages[0]) + 1):
    #     time.sleep(1)
    #     new_page = requests.get(BASE_URL + str(i), headers=HEADERS)
    #     new_tree = html.fromstring(new_page.content)
    #     asins = new_tree.xpath('//*[@data-asin]/@data-asin')
    #     products_asin.update(asins)
    #     # primes = new_tree.xpath('//*[@aria-label="Amazon Prime"]/@aria-label')
    #     # PRIME = soup.select_one('[aria-label="Amazon Prime"]')
    #     # PRIME = PRIME['aria-label']
    #     print("PAGE: " + str(i) + "\n")
    #
    # products_asin.remove('')
    #
    # with open("amazon_products.tsv", mode="w", encoding="utf-8") as out_file:
    #     tsv_writer = csv.writer(out_file, delimiter='\t')
    #     for asin in products_asin:
    #         prod_url = "https://www.amazon.it/dp/" + asin
    #         time.sleep(1)
    #         prod_page = requests.get(prod_url, headers=HEADERS)
    #         soup = BeautifulSoup(prod_page.content, "lxml")
    #         print(is_prime(soup))
    #         #tsv_writer.writerow([get_title(soup), get_price(soup), get_description(soup)])
