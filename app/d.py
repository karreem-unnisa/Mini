import pandas as pd

# Dataset based on the previous responses
data = [
    ["Home-based Craft Business", "Business", "Crafting, Marketing", "Low", "High", "High", "High"],
    ["Custom Sewing and Alterations", "Business", "Sewing, Tailoring", "Low", "Moderate", "Moderate", "High"],
    ["Crochet and Knit Accessories", "Business", "Crocheting, Knitting", "Low", "Moderate", "High", "High"],
    ["DIY Jewelry Making", "Business", "Crafting, Jewelry Design", "Low", "Moderate", "High", "High"],
    ["Etsy Store for Handmade Goods", "Business", "Crafting, E-Commerce", "Low", "High", "High", "High"],
    ["Upcycled Home Decor", "Business", "Crafting, DIY", "Low", "Moderate", "Moderate", "High"],
    ["Custom Clothing Design", "Business", "Sewing, Fashion Design", "Moderate", "High", "High", "Moderate"],
    ["Handmade Pottery Business", "Business", "Pottery, Crafting", "Moderate", "Moderate", "Moderate", "Moderate"],
    ["Macramé Home Decor", "Business", "Crafting, Macramé", "Low", "Moderate", "Moderate", "High"],
    ["Crochet Patterns Online Store", "Business", "Crocheting, Digital Product Creation", "Low", "Moderate", "Moderate", "High"],
    ["Candle Making Business", "Business", "Crafting, Product Creation", "Low", "Moderate", "Moderate", "High"],
    ["Online Sewing Classes", "Business", "Sewing, Teaching", "Low", "High", "High", "High"],
    ["DIY Wedding Decor Business", "Business", "Crafting, Event Planning", "Low", "Moderate", "Moderate", "Moderate"],
    ["Upcycling Old Clothes", "Business", "Sewing, DIY", "Low", "Moderate", "Moderate", "High"],
    ["Bespoke Embroidery Services", "Business", "Embroidery, Sewing", "Low", "Moderate", "Moderate", "Moderate"],
    ["Knitted Baby Clothes Business", "Business", "Knitting, Product Creation", "Low", "Moderate", "Moderate", "High"],
    ["Photography for Craft Products", "Business", "Photography, Marketing", "Low", "Moderate", "High", "High"],
    ["Scrapbooking and Card Making", "Business", "Crafting, DIY", "Low", "Low", "Moderate", "Moderate"],
    ["Custom Quilting Business", "Business", "Sewing, Quilting", "Low", "Moderate", "Moderate", "Moderate"],
    ["Crafting with Natural Materials", "Business", "Crafting, Sustainability", "Low", "Moderate", "Moderate", "High"],
    ["Home-Based Baking Business", "Business", "Baking, Cooking", "Moderate", "High", "High", "Moderate"],
    ["Selling Handmade Soap", "Business", "Soap Making, Crafting", "Low", "Moderate", "High", "High"],
    ["Creating Personalized Gifts", "Business", "Crafting, Personalization", "Low", "Moderate", "Moderate", "High"],
    ["Online Crochet Tutorials", "Business", "Crocheting, Teaching", "Low", "High", "High", "High"],
    ["Sewing for Kids Clothing", "Business", "Sewing, Fashion Design", "Low", "Moderate", "High", "Moderate"],
    ["Creating Handmade Wooden Crafts", "Business", "Woodworking, Crafting", "Moderate", "Moderate", "Moderate", "Low"],
    ["Natural Beauty Products Business", "Business", "DIY, Crafting", "Moderate", "High", "High", "Moderate"],
    ["Creating Handmade Bags and Accessories", "Business", "Crafting, Sewing", "Low", "Moderate", "Moderate", "High"],
    ["Knitted Scarves and Hats", "Business", "Knitting, Product Creation", "Low", "Moderate", "Moderate", "High"],
    ["Handmade Ceramic Products", "Business", "Pottery, Crafting", "Moderate", "Moderate", "Moderate", "Low"],
    ["Recycled Clothing for Fashion", "Business", "Sewing, Sustainability", "Low", "Moderate", "Moderate", "High"]
]

# Creating a DataFrame
df = pd.DataFrame(data, columns=["Business_Idea", "Category", "Skill_Required", "Initial_Investment", "Growth_Potential", "Market_Demand", "Flexibility"])

# Save the dataset to a CSV file at the desired path
file_path = r"D:\Mini-project\Mini\datasets\Women_Crafting_Business_Ideas.csv"
df.to_csv(file_path, index=False)

print(f"Dataset saved to {file_path}")
