label_map = {
    0: "T-Shirt",
    1: "Longsleeve",
    2: "Pants",
    3: "Shoes",
    4: "Shirt",
    5: "Dress",
    6: "Outwear",
    7: "Shorts",
    8: "Hat",
    9: "Skirt",
    10: "Polo",
    11: "Undershirt",
    12: "Blazer",
    13: "Hoodie",
    14: "Body",
    15: "Top",
    16: "Blouse",
    17: "Not sure",
    18: "Other",
    19: "Skip",
}


def get_label_id(label):
    for key, value in label_map.items():
        if value == label:
            return key
    return None


def get_label_name(label_id):
    return label_map[label_id]
