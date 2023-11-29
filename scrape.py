import requests
from bs4 import BeautifulSoup


def scrape_main_panel(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Base URL to prepend to relative links
    base_url = 'https://lifeshiftplatform.com'

    # Find all <li> tags with class "item" and extract the URLs
    # Assuming the link is contained within an <a> tag inside the <li> tag
    links = []
    for li in soup.find_all('li', class_='item'):
        a_tag = li.find('a')
        if a_tag and 'href' in a_tag.attrs:
            link = a_tag['href']
            # Check if the link is relative and prepend the base URL
        if not link.startswith('http'):
            if link.startswith('/member/'):
                link = base_url + link
                links.append(link)
    return links

def scrape_individual_page(url):
    print(url)
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract the name of the person using the 'name' class
    name = soup.find('h1', class_='name').get_text(strip=True)
    print(name)

    # Extract the details. Since it's after a <br> tag within a <div class="detail">,
    # you can find the <div>, then get the <p> tag text content.
    detail_div = soup.find('div', class_='detail')
    details = detail_div.find('p').get_text(strip=True)

    # Assuming the details you need come after a 'br' tag, and that there is plain text following the 'br' tag.
    # Below code splits the text by 'br' tag and gets the second part (index 1)
    # If there are multiple 'br' tags and you need text after the first 'br', you would adjust indexing accordingly.
    detail_parts = details.split('<br>', 1)
    if len(detail_parts) > 1:
        detail_text = detail_parts[1]
    else:
        detail_text = detail_parts[0]  # Fallback in case there is no '<br>'
    print("name: ", name, "details: :", detail_text)
    return {'name': name, 'details': detail_text}

# Replace this with the URL of your main panel page
main_panel_url = 'https://lifeshiftplatform.com/member'
individual_links = scrape_main_panel(main_panel_url)
print("idvLink: ", len(individual_links))

all_persons_info = []
for link in individual_links:
    person_info = scrape_individual_page(link)
    all_persons_info.append(person_info)

# Now all_persons_info contains the information for each person
