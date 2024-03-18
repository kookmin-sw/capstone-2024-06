import requests


headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


def get_desks(page=1):
    api_url = "https://ohou.se/store/category.json"
    params = {
        "v": 2,
        "category_id": 10150004,
        "page": page,
        "per": 24
    }

    response = requests.get(api_url, params=params, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data. Status code: {response.status_code}")

    desks = []
    fetched_desks = response.json()["productions"]
    for fetched_desk in fetched_desks:
        desks.append(
            {
                "id": fetched_desk["id"],
                "name": fetched_desk["name"],
                "image_url": fetched_desk["original_image_url"]
            }
        )
    return desks


def get_styling_shots(desk_id, page=1):
    api_url = f"https://ohou.se/productions/{desk_id}/used_card.json"
    params = {
        "page": page,
        "per": 48
    }

    response = requests.get(api_url, params=params, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data. Status code: {response.status_code}")
    
    styling_shots = []
    fetched_shots = response.json()["cards"]
    for fetched_shot in fetched_shots:
        styling_shots.append(
            {
                "id": fetched_shot["id"],
                "image_url": fetched_shot["original_image_url"]
            }
        )
    return styling_shots


def get_tags(styling_shot_id):
    api_url = f"https://contents.ohou.se/api/cards/{styling_shot_id}"
    response = requests.get(api_url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data. Status code: {response.status_code}")
    
    tags = []
    fetched_tags = response.json()["data"]["card"]["tags"]
    for fetched_tag in fetched_tags:
        if fetched_tag["type"] != "product":
            continue

        tags.append(
            {
                "id": fetched_tag["product"]["id"],
                "image_url": fetched_tag["product"]["imageUrl"],
                "name": fetched_tag["product"]["name"],
                "price": fetched_tag["product"]["price"],
                "x": fetched_tag["positionX"],
                "y": fetched_tag["positionY"]
            }
        )
    return tags


if __name__ == "__main__":


    from PIL import Image
    from io import BytesIO
    from pprint import pprint
    import re


    def show_image_from_url(url):
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        image.show()


    def sanitize_filename(filename):
        sanitized_filename = re.sub(r'[\/\\\:\*\?\"\<\>\|]', '_', filename)
        sanitized_filename = ''.join(c for c in sanitized_filename if c.isprintable())
        return sanitized_filename.replace(" ", "_")
    

    def download_image(url, file_name, download_folder):
        response = requests.get(url)
        if response.status_code == 200:
            with open(f"./{download_folder}/{file_name}.png", 'wb') as f:
                f.write(response.content)

    download_folder = "image"

    for page in range(1, 100):
        print(page)
        desks = get_desks(page=page)
        for desk in desks:
            download_image(desk["image_url"], sanitize_filename(desk["name"]), download_folder)
    # target_desk = desks[0]
    # show_image_from_url(target_desk["image_url"])

    # styling_shots = get_styling_shots(target_desk["id"], page=1)
    # pprint(styling_shots)
    # target_styling_shot = styling_shots[0]
    # show_image_from_url(target_styling_shot["image_url"])

    # tags = get_tags(target_styling_shot["id"])
    # pprint(tags)
