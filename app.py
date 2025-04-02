from flask import Flask, request, jsonify, send_file
import requests
from PIL import Image, ImageDraw, ImageChops
import io
import glob
import os
import base64
from skimage.metrics import structural_similarity as ssim
from collections import OrderedDict
from priority_cache_manager import PriorityCacheManager
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import ImageEnhance


app = Flask(__name__)

REFERENCE_IMAGE_SIZE = 2810
CONFIDENCE_THRESHOLD_CIRCLE = 0.85
CONFIDENCE_THRESHOLD_DIAMOND = 0.89

RAW_COLOR_MAP = {
    "#F156FF": "decision",
    "#2DB38F": "easy",
    "#ECD982": "medium",
    "#F07E5F": "hard",
    "#9843C6": "portal",
    "#CA3B5F": "arrival",
    "#D7995B": "bronze door",
    "#EAE9E8": "silver door",
    "#FFDF33": "gold door",
    "#6D6DE5": "shop",
    "#697785": "time lock",
    "#E58F16": "boss"
}

COLOR_MAP = {k.upper(): v for k, v in RAW_COLOR_MAP.items()}

islandCenters = [
    { "bgX": 1404.5, "bgY": 343.5 },
    { "bgX": 1140.5, "bgY": 607.5 },
    { "bgX": 1668.5, "bgY": 607.5 },
    { "bgX": 876.5, "bgY": 871.5 },
    { "bgX": 1404.5, "bgY": 871.5 },
    { "bgX": 1932.5, "bgY": 871.5 },
    { "bgX": 612.5, "bgY": 1135.5 },
    { "bgX": 1140.5, "bgY": 1135.5 },
    { "bgX": 1668.5, "bgY": 1135.5 },
    { "bgX": 2196.5, "bgY": 1136.0 },
    { "bgX": 348.5, "bgY": 1399.5 },
    { "bgX": 876.5, "bgY": 1399.5 },
    { "bgX": 1404.5, "bgY": 1399.5 },
    { "bgX": 1932.5, "bgY": 1399.0 },
    { "bgX": 2460.5, "bgY": 1400.0 },
    { "bgX": 612.5, "bgY": 1663.5 },
    { "bgX": 1140.5, "bgY": 1663.0 },
    { "bgX": 1668.5, "bgY": 1663.5 },
    { "bgX": 2196.5, "bgY": 1663.0 },
    { "bgX": 876.5, "bgY": 1927.5 },
    { "bgX": 1404.5, "bgY": 1927.5 },
    { "bgX": 1932.5, "bgY": 1928.0 },
    { "bgX": 1140.5, "bgY": 2191.5 },
    { "bgX": 1668.5, "bgY": 2192.0 },
    { "bgX": 1404.5, "bgY": 2455.5 }
]
combatTypePoints = [
    {"bossX": 1406, "bossY": 170, "minionX": 1404, "minionY": 494},
    {"bossX": 1142, "bossY": 434, "minionX": 1140, "minionY": 758},
    {"bossX": 1670, "bossY": 434, "minionX": 1668, "minionY": 758},
    {"bossX": 878, "bossY": 698, "minionX": 876, "minionY": 1022},
    {"bossX": 1406, "bossY": 698, "minionX": 1404, "minionY": 1022},
    {"bossX": 1934, "bossY": 698, "minionX": 1932, "minionY": 1022},
    {"bossX": 614, "bossY": 962, "minionX": 612, "minionY": 1286},
    {"bossX": 1142, "bossY": 962, "minionX": 1140, "minionY": 1286},
    {"bossX": 1670, "bossY": 962, "minionX": 1668, "minionY": 1286},
    {"bossX": 2199, "bossY": 963, "minionX": 2196, "minionY": 1287},
    {"bossX": 350, "bossY": 1225, "minionX": 348, "minionY": 1550},
    {"bossX": 878, "bossY": 1225, "minionX": 876, "minionY": 1550},
    {"bossX": 1406, "bossY": 1226, "minionX": 1404, "minionY": 1550},
    {"bossX": 1934, "bossY": 1226, "minionX": 1932, "minionY": 1549},
    {"bossX": 2462, "bossY": 1226, "minionX": 2460, "minionY": 1550},
    {"bossX": 614, "bossY": 1489, "minionX": 612, "minionY": 1814},
    {"bossX": 1142, "bossY": 1489, "minionX": 1140, "minionY": 1813},
    {"bossX": 1670, "bossY": 1489, "minionX": 1668, "minionY": 1814},
    {"bossX": 2199, "bossY": 1489, "minionX": 2196, "minionY": 1813},
    {"bossX": 878, "bossY": 1753, "minionX": 876, "minionY": 2077},
    {"bossX": 1406, "bossY": 1753, "minionX": 1404, "minionY": 2077},
    {"bossX": 1934, "bossY": 1754, "minionX": 1932, "minionY": 2079},
    {"bossX": 1142, "bossY": 2017, "minionX": 1140, "minionY": 2340},
    {"bossX": 1670, "bossY": 2018, "minionX": 1668, "minionY": 2341},
    {"bossX": 1406, "bossY": 2281, "minionX": 1404, "minionY": 2603}
]
arrowPointsA = [
    'x',
    'x',
    [[1558, 497], [1515, 454]],
    'x',
    [[1294, 761], [1251, 718]],
    [[1822, 761], [1779, 718]],
    'x',
    [[1030, 1025], [987, 982]],
    [[1558, 1025], [1515, 982]],
    [[2086, 1025], [2043, 982]],
    'x',
    [[766, 1289], [723, 1246]],
    [[1294, 1289], [1251, 1246]],
    [[1822, 1289], [1779, 1246]],
    [[2350, 1289], [2307, 1246]],
    [[502, 1553], [459, 1510]],
    [[1030, 1553], [987, 1510]],
    [[1558, 1553], [1515, 1510]],
    [[2086, 1553], [2043, 1510]],
    [[766, 1817], [723, 1774]],
    [[1294, 1817], [1251, 1774]],
    [[1822, 1817], [1779, 1774]],
    [[1030, 2081], [987, 2038]],
    [[1558, 2081], [1515, 2038]],
    [[1294, 2345], [1251, 2302]]
]
arrowPointsD = [
    'x',
    [[1251, 497], [1294, 454]],
    'x',
    [[987, 761], [1030, 718]],
    [[1515, 761], [1558, 718]],
    'x',
    [[723, 1025], [766, 982]],
    [[1251, 1025], [1294, 982]],
    [[1779, 1025], [1822, 982]],
    'x',
    [[459, 1289], [502, 1246]],
    [[987, 1289], [1030, 1246]],
    [[1515, 1289], [1558, 1246]],
    [[2043, 1289], [2086, 1246]],
    'x',
    [[723, 1553], [766, 1510]],
    [[1251, 1553], [1294, 1510]],
    [[1779, 1553], [1822, 1510]],
    [[2307, 1553], [2350, 1510]],
    [[987, 1817], [1030, 1774]],
    [[1515, 1817], [1558, 1774]],
    [[2043, 1817], [2086, 1774]],
    [[1251, 2081], [1294, 2038]],
    [[1779, 2081], [1822, 2038]],
    [[1515, 2345], [1558, 2302]]
]

icon_points = [
            { "leftX": 1304, "leftY": 343, "rightX": 1504, "rightY": 343 },
            { "leftX": 1040, "leftY": 607, "rightX": 1240, "rightY": 607 },
            { "leftX": 1570, "leftY": 607, "rightX": 1770, "rightY": 607 },
            { "leftX": 776, "leftY": 871, "rightX": 976, "rightY": 871 },
            { "leftX": 1304, "leftY": 871, "rightX": 1504, "rightY": 871 },
            { "leftX": 1832, "leftY": 871, "rightX": 2032, "rightY": 871 },
            { "leftX": 512, "leftY": 1135, "rightX": 712, "rightY": 1135 },
            { "leftX": 1040, "leftY": 1135, "rightX": 1240, "rightY": 1135 },
            { "leftX": 1570, "leftY": 1135, "rightX": 1770, "rightY": 1135 },
            { "leftX": 2098, "leftY": 1135, "rightX": 2298, "rightY": 1135 },
            { "leftX": 248, "leftY": 1399, "rightX": 448, "rightY": 1399 },
            { "leftX": 776, "leftY": 1399, "rightX": 976, "rightY": 1399 },
            { "leftX": 1304, "leftY": 1399, "rightX": 1504, "rightY": 1399 },
            { "leftX": 1832, "leftY": 1399, "rightX": 2032, "rightY": 1399 },
            { "leftX": 2360, "leftY": 1399, "rightX": 2560, "rightY": 1399 },
            { "leftX": 512, "leftY": 1663, "rightX": 712, "rightY": 1663 },
            { "leftX": 1040, "leftY": 1663, "rightX": 1240, "rightY": 1663 },
            { "leftX": 1570, "leftY": 1663, "rightX": 1770, "rightY": 1663 },
            { "leftX": 2098, "leftY": 1663, "rightX": 2298, "rightY": 1663 },
            { "leftX": 776, "leftY": 1927, "rightX": 976, "rightY": 1927 },
            { "leftX": 1304, "leftY": 1927, "rightX": 1504, "rightY": 1927 },
            { "leftX": 1832, "leftY": 1927, "rightX": 2032, "rightY": 1927 },
            { "leftX": 1040, "leftY": 2191, "rightX": 1240, "rightY": 2191 },
            { "leftX": 1570, "leftY": 2191, "rightX": 1770, "rightY": 2191 },
            { "leftX": 1304, "leftY": 2455, "rightX": 1504, "rightY": 2455 }
        ]

door_categories = {"bronze door", "silver door", "door", "gold door", "time lock"}
symbol_categories = {"portal", "arrival", "shop"}
ssim_categories = {"decision"}
image_categories = {"battle", "boss"}

def preprocess_crop(crop, size=(118, 118)):
    enhancer = ImageEnhance.Contrast(crop.convert("L"))
    boosted = enhancer.enhance(1.5)  # increase contrast
    return boosted.resize(size)

def enhance_contrast(img, factor=1.5):
    return ImageEnhance.Contrast(img).enhance(factor)

def hex_to_rgb(hex_color):
    return tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))

def color_distance(c1, c2):
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5

def closest_color(pixel):
    closest_hex = None
    closest_dist = float("inf")

    for hex_code in COLOR_MAP.keys():
        color_rgb = hex_to_rgb(hex_code)
        dist = color_distance(pixel, color_rgb)
        if dist < closest_dist:
            closest_dist = dist
            closest_hex = hex_code

    if closest_dist < 20:
        return closest_hex  # fuzzy match accepted

    # No acceptable match — return exact color as hex
    return "#{:02X}{:02X}{:02X}".format(pixel[0], pixel[1], pixel[2])

priority_cache = PriorityCacheManager(original_capacity=6, scaled_capacity=6)

CANNOT_BE_MINION_COLORS = {
    "#2DB38F",  # Easy
    "#ECD982",  # Medium
    "#F07E5F"   # Hard
}

MONSTER_HEX = "#262B34"  # Dark fallback (Monster)
MONSTER_THRESHOLD = 10  # Allow fuzzy match within distance 10
def is_monster_color(pixel_hex):
    pixel_rgb = hex_to_rgb(pixel_hex)
    monster_rgb = hex_to_rgb(MONSTER_HEX)
    return color_distance(pixel_rgb, monster_rgb) <= MONSTER_THRESHOLD

def is_minion_color(pixel_hex):
    if pixel_hex.upper() in CANNOT_BE_MINION_COLORS or is_monster_color(pixel_hex):
        return False
    return True

def download_image(image_url):
    cached_image = priority_cache.get_original(image_url)
    if cached_image:
        print(f"Using cached original for {image_url}")
        return cached_image

    response = requests.get(image_url)
    if response.status_code != 200:
        raise Exception(f"Cloudinary returned error {response.status_code}.")

    img = Image.open(io.BytesIO(response.content)).convert("RGBA")
    priority_cache.store_original(image_url, img)  # Auto-tracks batch/type if present
    return img

def get_image_scale(image):
    return image.width / REFERENCE_IMAGE_SIZE

def preload_er_icon_templates(directories, er_scaled_size=118):
    templates = {}
    for directory in directories:
        for file in glob.glob(f"{directory}/*.png"):
            img_pil = Image.open(file).convert("L")
            img_np = np.array(img_pil)  # full-size 200px

            scaled_img = img_pil.resize((er_scaled_size, er_scaled_size))
            scaled_array = np.array(scaled_img)

            name = os.path.basename(file).split(".")[0]
            templates[name] = {
                "original": img_np,
                "er_scaled": scaled_array
            }
    
    print(f"[TEMPLATE LOAD] Loaded template: {name} from {file}")
    return templates



icon_templates = preload_er_icon_templates(["iconsER"], er_scaled_size=118)

def image_similarity_ssim(img1, img2):
    img1_gray = np.array(img1.convert("L"))
    img2_gray = np.array(img2.convert("L").resize(img1.size))
    score, _ = ssim(img1_gray, img2_gray, full=True)
    return score

def find_best_match_icon(preprocessed_img, threshold=0.85):
    img_array = np.array(preprocessed_img)
    best_score = -1
    best_name = "other"

    for name in sorted(icon_templates.keys()):
        template_array = icon_templates[name]["er_scaled"]
        score = ssim(img_array, template_array)
        if score > best_score:
            best_score = score
            best_name = name

    return {
        "label": best_name if best_score >= threshold else "other",
        "score": best_score
    }

def best_shifted_match(x, y, image, threshold=0.85):
    scale = get_image_scale(image)
    scaled_x = int(x * scale)
    scaled_y = int(y * scale)
    radius = int(100 * scale)

    best_result = {"label": "other", "score": -1, "base64": ""}
    best_crop = None

    def crop_and_match(dx, dy):
        cx, cy = scaled_x + dx, scaled_y + dy
        crop_coords = [
            (cx, cy - radius), (cx - radius, cy),
            (cx, cy + radius), (cx + radius, cy)
        ]
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).polygon(crop_coords, fill=255)
        cropped = Image.composite(image, Image.new("RGBA", image.size, (0, 0, 0, 0)), mask).crop(
            (cx - radius, cy - radius, cx + radius, cy + radius)
        )
        pre = preprocess_crop(cropped)  # grayscale + contrast + resize
        match = find_best_match_icon(pre, threshold)
        return match, cropped

    for dx in [-2, -1, 0, 1, 2]:
        for dy in [-2, -1, 0, 1, 2]:
            match, cropped = crop_and_match(dx, dy)
            if match["score"] > best_result["score"]:
                best_result = match
                best_crop = cropped

    if best_crop:
        best_result["base64"] = image_to_base64(best_crop)

    return best_result


def image_to_base64(img):
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

@app.route('/extract_color', methods=['GET'])
def extract_color():
    image_url = request.args.get("image_url")
    x, y = int(request.args.get("x", 0)), int(request.args.get("y", 0))

    try:
        image = download_image(image_url)
        scale_factor = get_image_scale(image)
        pixel = image.getpixel((int(x * scale_factor), int(y * scale_factor)))

        color_result = closest_color(pixel)
        return jsonify({"hex": color_result})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/check_minion', methods=['GET'])
def check_minion():
    image_url = request.args.get("image_url")
    x, y = int(request.args.get("x", 0)), int(request.args.get("y", 0))

    try:
        image = download_image(image_url)
        scale_factor = get_image_scale(image)
        pixel = image.getpixel((int(x * scale_factor), int(y * scale_factor)))
        pixel_hex = "#{:02X}{:02X}{:02X}".format(pixel[0], pixel[1], pixel[2])

        return jsonify({"minion": is_minion_color(pixel_hex)})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/extract_all_categories', methods=['POST'])
def extract_all_categories():
    try:
        data = request.get_json()
        image_url = data.get("image_url")
        customCenters = data.get("islandCenters")
        centers_to_use = customCenters if customCenters else islandCenters

        if not image_url:
            return jsonify({"error": "Missing image_url"}), 400

        img = download_image(image_url)
        scale = get_image_scale(img)

        def process_island(i, center):
            try:
                x_scaled = int(center["bgX"] * scale)
                y_scaled = int(center["bgY"] * scale)
                pixel = img.getpixel((x_scaled, y_scaled))
                matched_hex = closest_color(pixel[:3])
                island_type = COLOR_MAP.get(matched_hex, "Void") if matched_hex else "Void"

                boss_x = int(combatTypePoints[i]["bossX"] * scale)
                boss_y = int(combatTypePoints[i]["bossY"] * scale)
                boss_pixel = img.getpixel((boss_x, boss_y))
                boss_hex = "#{:02X}{:02X}{:02X}".format(*boss_pixel[:3])

                minion_x = int(combatTypePoints[i]["minionX"] * scale)
                minion_y = int(combatTypePoints[i]["minionY"] * scale)
                minion_pixel = img.getpixel((minion_x, minion_y))
                minion_hex = "#{:02X}{:02X}{:02X}".format(*minion_pixel[:3])

                combat_type_helper = "None"
                if boss_hex.upper() == "#E58F16":
                    combat_type_helper = "boss"
                elif is_minion_color(minion_hex):
                    combat_type_helper = "minion"

                lower_type = island_type.lower()
                if lower_type in ["easy", "medium", "hard"]:
                    category = combat_type_helper if combat_type_helper != "None" else "battle"
                elif lower_type == "decision":
                    category = "decision"
                elif lower_type == "shop":
                    category = "shop"
                elif lower_type in ["portal", "arrival"]:
                    category = "portal"
                elif lower_type in ["bronze door", "silver door", "gold door", "time lock"]:
                    category = "door"

                else:
                    category = "Void"

                print(f"Island {i+1} RGB: {pixel}, Closest Hex: {matched_hex}, Matched Type: {island_type}")

                return {
                    "index": i + 1,
                    "island_type": island_type,
                    "category": category
                }

            except Exception as inner_e:
                print(f"Error processing island {i+1}: {str(inner_e)}")
                return {"index": i + 1, "island_type": "Void", "category": "Void"}

        results = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_island, i, center) for i, center in enumerate(centers_to_use)]
            for f in as_completed(futures):
                results.append(f.result())

        results.sort(key=lambda x: x["index"])
        return jsonify({"island_data": results})

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/crop_circle', methods=['POST'])
def crop_circle():
    data = request.get_json()
    image_url, x, y = data.get("image_url"), int(data.get("x")), int(data.get("y"))

    try:
        image = download_image(image_url)
        scale_factor = get_image_scale(image)
        x_scaled, y_scaled = int(x * scale_factor), int(y * scale_factor)
        radius = int(24 * scale_factor)

        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((x_scaled - radius, y_scaled - radius, x_scaled + radius, y_scaled + radius), fill=255)

        cropped_img = Image.composite(image, Image.new("RGBA", image.size, (0,0,0,0)), mask).crop(
            (x_scaled - radius, y_scaled - radius, x_scaled + radius, y_scaled + radius))

        label = find_best_match_icon(cropped_img, CONFIDENCE_THRESHOLD_CIRCLE)
        output = io.BytesIO()
        cropped_img.save(output, format="PNG")
        image_base64 = base64.b64encode(output.getvalue()).decode("utf-8")
        priority_cache.store_scaled(image_url, 48, cropped_img)  # assuming 48px target scale for small crops

        return jsonify({"label": label, "base64": image_base64})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/crop_diamond', methods=['POST'])
def crop_diamond():
    data = request.get_json()
    image_url, x, y = data.get("image_url"), int(data.get("x")), int(data.get("y"))

    try:
        image = download_image(image_url)
        scale_factor = get_image_scale(image)
        scaled_x, scaled_y = int(x * scale_factor), int(y * scale_factor)
        radius = int(100 * scale_factor)

        crop_coords = [
            (scaled_x, scaled_y - radius), (scaled_x - radius, scaled_y),
            (scaled_x, scaled_y + radius), (scaled_x + radius, scaled_y)
        ]

        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).polygon(crop_coords, fill=255)

        cropped_img = Image.composite(image, Image.new("RGBA", image.size, (0, 0, 0, 0)), mask).crop(
            (scaled_x - radius, scaled_y - radius, scaled_x + radius, scaled_y + radius)
        )

        # Get label from icon matching
        label = find_best_match_icon(cropped_img, CONFIDENCE_THRESHOLD_DIAMOND)

        output = io.BytesIO()
        cropped_img.save(output, format="PNG")
        image_base64 = base64.b64encode(output.getvalue()).decode("utf-8")

        scale = image.width / 2810
        priority_cache.store_scaled(image_url, scale, cropped_img)

        return jsonify({
            "base64": image_base64,
            "label": label
        })

    except Exception as e:
        return jsonify({"error": str(e)})

def crop_diamond_to_file(image_url, x, y, output_path="diamond_crop.png"):
    try:
        img = download_image(image_url)
        scale_factor = get_image_scale(img)
        scaled_x, scaled_y = int(x * scale_factor), int(y * scale_factor)
        radius = int(100 * scale_factor)

        crop_coords = [
            (scaled_x, scaled_y - radius), (scaled_x - radius, scaled_y),
            (scaled_x, scaled_y + radius), (scaled_x + radius, scaled_y)
        ]

        mask = Image.new("L", img.size, 0)
        ImageDraw.Draw(mask).polygon(crop_coords, fill=255)

        cropped_img = Image.composite(img, Image.new("RGBA", img.size, (0, 0, 0, 0)), mask).crop(
            (scaled_x - radius, scaled_y - radius, scaled_x + radius, scaled_y + radius)
        )

        cropped_img.save(output_path)
        return f"Saved diamond crop to {output_path}"
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/crop_small_diamond', methods=['POST'])
def crop_small_diamond():
    data = request.get_json()
    image_url, x, y = data.get("image_url"), int(data.get("x")), int(data.get("y"))

    try:
        image = download_image(image_url)
        scale_factor = get_image_scale(image)
        x_scaled, y_scaled = int(x * scale_factor), int(y * scale_factor)
        offset = int(32 * scale_factor)

        crop_coords = [
            (x_scaled, y_scaled - offset), (x_scaled - offset, y_scaled),
            (x_scaled, y_scaled + offset), (x_scaled + offset, y_scaled)
        ]

        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).polygon(crop_coords, fill=255)

        cropped_img = Image.composite(image, Image.new("RGBA", image.size, (0,0,0,0)), mask).crop(
            (x_scaled - offset, y_scaled - offset, x_scaled + offset, y_scaled + offset))

        output = io.BytesIO()
        cropped_img.save(output, format="PNG")
        image_base64 = base64.b64encode(output.getvalue()).decode("utf-8")

        return jsonify({"image_base64": image_base64})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/arrow_check_bulk', methods=['POST'])
def arrow_check_bulk():
    try:
        data = request.get_json()
        image_url = data.get("image_url")

        if not image_url:
            return jsonify({"error": "Missing image_url"}), 400

        pointsA = arrowPointsA
        pointsD = arrowPointsD

        img = download_image(image_url)
        scale = get_image_scale(img)

        def check_color(x, y):
            scaled_x, scaled_y = int(x * scale), int(y * scale)
            pixel = img.getpixel((scaled_x, scaled_y))
            hex_color = "#{:02X}{:02X}{:02X}".format(*pixel[:3])
            accepted_colors = {
                "#F156FF", "#FFFFFF", "#2DB38F", "#ECD982", "#E5E4E2",
                "#FFD700", "#CD7F32", "#445566", "#F07E5F", "#EAE9E8"
            }
            return "arrow" if hex_color.upper() in accepted_colors else "no"

        def process_arrow_pair(entry):
            if entry == "x":
                return ["skip", "skip"]
            else:
                (x1, y1), (x2, y2) = entry
                result1 = check_color(x1, y1)
                result2 = check_color(x2, y2)
                return [result1, result2]

        results = {
            "A": [process_arrow_pair(entry) for entry in pointsA],
            "D": [process_arrow_pair(entry) for entry in pointsD]
        }

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/crop_all_decision_icons', methods=['POST'])
def crop_all_decision_icons():
    try:
        data = request.get_json()
        image_url = data.get("image_url")
        categories = data.get("categories", [])
        if not image_url:
            return jsonify({"error": "Missing image_url"}), 400
        if not categories or len(categories) != 25:
            return jsonify({"error": "categories must be a 25-item list"}), 400

        img = download_image(image_url)

        def crop_diamond_scaled(x, y):
            scale = get_image_scale(img)
            scaled_x = int(x * scale)
            scaled_y = int(y * scale)
            radius = int(100 * scale)

            crop_coords = [
                (scaled_x, scaled_y - radius), (scaled_x - radius, scaled_y),
                (scaled_x, scaled_y + radius), (scaled_x + radius, scaled_y)
            ]

            mask = Image.new("L", img.size, 0)
            ImageDraw.Draw(mask).polygon(crop_coords, fill=255)

            return Image.composite(img, Image.new("RGBA", img.size, (0, 0, 0, 0)), mask).crop(
                (scaled_x - radius, scaled_y - radius, scaled_x + radius, scaled_y + radius)
            )

        def process_icon(idx):
            point = icon_points[idx]
            category = categories[idx].strip().lower()

            left_result = {"id": f"L{idx+1}", "label": "", "base64": ""}
            right_result = {"id": f"R{idx+1}", "label": "", "base64": ""}

            try:
                if category in ssim_categories:
                    left_match = best_shifted_match(point["leftX"], point["leftY"], img)
                    right_match = best_shifted_match(point["rightX"], point["rightY"], img)
                    left_result["label"] = left_match["label"]
                    left_result["base64"] = left_match["base64"]
                    right_result["label"] = right_match["label"]
                    right_result["base64"] = right_match["base64"]


                elif category in image_categories:
                    left_crop = crop_diamond_scaled(point["leftX"], point["leftY"])
                    right_crop = crop_diamond_scaled(point["rightX"], point["rightY"])

                    left_result["label"] = category
                    right_result["label"] = category
                    left_result["base64"] = image_to_base64(left_crop)
                    right_result["base64"] = image_to_base64(right_crop)

                elif category in door_categories:
                    left_result["label"] = "𓉞"
                    right_result["label"] = "𓉞"

                elif category in symbol_categories:
                    left_result["label"] = "⋆₊˚⊹"
                    right_result["label"] = "࿔⋆"

            except Exception as e:
                print(f"Error processing icon {idx+1}: {str(e)}")

            return {"left": left_result, "right": right_result}



        results = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_icon, idx) for idx in range(25)]
            for f in as_completed(futures):
                results.append(f.result())

        results.sort(key=lambda r: int(r['left']['id'][1:]))
        return jsonify({"icons": results})

    except Exception as e:
        print("ERROR in crop_all_decision_icons:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/debug_decision_icon_labels', methods=['POST'])
def debug_decision_icon_labels():
    try:
        data = request.get_json()
        image_url = data.get("image_url")
        categories = data.get("categories", [])
        if not image_url:
            return jsonify({"error": "Missing image_url"}), 400
        if not categories or len(categories) != 25:
            return jsonify({"error": "categories must be a 25-item list"}), 400

        img = download_image(image_url)
        scale = get_image_scale(img)

        def crop_diamond_scaled(x, y):
            scaled_x, scaled_y = int(x * scale), int(y * scale)
            radius = int(100 * scale)
            crop_coords = [
                (scaled_x, scaled_y - radius), (scaled_x - radius, scaled_y),
                (scaled_x, scaled_y + radius), (scaled_x + radius, scaled_y)
            ]
            mask = Image.new("L", img.size, 0)
            ImageDraw.Draw(mask).polygon(crop_coords, fill=255)
            cropped = Image.composite(img, Image.new("RGBA", img.size, (0,0,0,0)), mask).crop(
                (scaled_x - radius, scaled_y - radius, scaled_x + radius, scaled_y + radius)
            )
            return cropped

        debug_output = []

        for idx in range(25):
            point = icon_points[idx]
            category = categories[idx].strip().lower()

            row = {"index": idx + 1, "category": category}

            if category == "decision":
                left_match = best_shifted_match(point["leftX"], point["leftY"], img)
                right_match = best_shifted_match(point["rightX"], point["rightY"], img)

                row.update({
                    "left_label": left_match["label"],
                    "left_score": left_match["score"],
                    "right_label": right_match["label"],
                    "right_score": right_match["score"]
                })

            else:
                row.update({
                    "left_label": "(skipped)",
                    "right_label": "(skipped)"
                })

            debug_output.append(row)

        return jsonify(debug_output)

    except Exception as e:
        print("ERROR in debug_decision_icon_labels:", str(e))
        return jsonify({"error": str(e)}), 500



@app.route('/crop_diamond_to_file', methods=['POST'])
def crop_diamond_to_file_route():
    try:
        data = request.get_json()
        image_url = data.get("image_url")
        x = int(data.get("x"))
        y = int(data.get("y"))
        output_path = data.get("output_path", "diamond_crop.png")

        result = crop_diamond_to_file(image_url, x, y, output_path)
        return jsonify({"message": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/debug_icon_at_point', methods=['POST'])
def debug_icon_at_point():
    try:
        data = request.get_json()
        image_url = data.get("image_url")
        x = int(data.get("x"))
        y = int(data.get("y"))
        threshold = float(data.get("threshold", 0.85))

        if not image_url:
            return jsonify({"error": "Missing image_url"}), 400

        img = download_image(image_url)
        best = best_shifted_match(x, y, img, threshold)

        return jsonify({
            "image_url": image_url,
            "x": x,
            "y": y,
            "best_label": best["label"],
            "best_score": best["score"],
            "base64": best["base64"]
        })

    except Exception as e:
        print("ERROR in debug_icon_at_point:", str(e))
        return jsonify({"error": str(e)}), 500
    
@app.route('/status', methods=['GET'])
def status():
    return jsonify(priority_cache.get_cache_status())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), threaded=True)



